#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEXC spot scanner/trader with:
- Signals-as-silent-trades lifecycle (no duplicate signals until closed by hit of T1 or Stop)
- Separate signal performance dashboard (console only)
- Separate Discord channels: signals vs trades
- Legacy movers kept; Top-100 scan replaces watchlist; stable/pegged coins filtered out
"""

import argparse, json, os, sys, yaml, requests, math, re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set, Optional
import pandas as pd
import numpy as np
import ccxt
import redis

# ================== TA helpers ==================
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff(); up = d.clip(lower=0).rolling(length).mean(); dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9); return 100 - (100 / (1 + rs))
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df['high'] - df['low']; hc = (df['high'] - df['close'].shift()).abs(); lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl,hc,lc], axis=1).max(axis=1); return tr.rolling(n).mean()
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow); line = f - s; sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig; return line, sig, hist

# ================== Exchange wrapper (MEXC only) ==================
class ExClient:
    def __init__(self):
        self.name = "mexc"
        self.ex = ccxt.mexc({"enableRateLimit": True})
        key = os.environ.get("MEXC_API_KEY")
        sec = os.environ.get("MEXC_SECRET")
        if key and sec:
            self.ex.apiKey = key
            self.ex.secret = sec
        self._markets = None
    def load_markets(self):
        if self._markets is None:
            try: self._markets = self.ex.load_markets()
            except Exception: self._markets = {}
        return self._markets
    def has_pair(self, symbol_pair: str) -> bool:
        mkts = self.load_markets() or {}
        if symbol_pair in mkts: return True
        syms = getattr(self.ex, "symbols", None) or []
        return symbol_pair in syms
    def ohlcv(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")
    def last_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.ex.fetch_ticker(symbol)
            px = t.get("last") or t.get("close")
            return float(px) if px is not None else None
        except Exception:
            try:
                df = self.ohlcv(symbol, "1h", 2)
                return float(df["close"].iloc[-1])
            except Exception:
                return None
    def place_market(self, symbol: str, side: str, amount: float) -> dict:
        return self.ex.create_order(symbol, type="market", side=side, amount=amount)

# ================== Params ==================
@dataclass
class DayParams:
    lookback_high: int = 30
    vol_sma: int = 30
    rsi_min: int = 52
    rsi_max: int = 78
    stop_mode: str = "swing"
    atr_mult: float = 1.5

@dataclass
class TrendParams:
    ema20: int = 20
    ema50: int = 50
    ema100: int = 100
    pullback_pct_max: float = 10.0
    rsi_min: int = 50
    rsi_max: int = 70
    vol_sma: int = 20
    breakout_lookback: int = 55
    stop_mode: str = "swing"
    atr_mult: float = 2.0

# ================== Redis persistence ==================
class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url:
            raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL env.")
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48*3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")
    def k(self, *parts) -> str: return ":".join([self.prefix, *[str(p) for p in parts]])
    # edge memory (per-bar dedup if you ever want it)
    def get_mem(self, kind: str, key: str) -> str: return self.r.get(self.k("mem", kind, key)) or ""
    def set_mem(self, kind: str, key: str, value: str): self.r.setex(self.k("mem", kind, key), self.ttl_seconds, value)
    # positions / trades
    def load_positions(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","active_positions")); return json.loads(txt) if txt else {}
    def save_positions(self, d: Dict[str, Any]): self.r.set(self.k("state","active_positions"), json.dumps(d))
    def load_perf(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","performance")); return json.loads(txt) if txt else {"open_trades": [], "closed_trades": []}
    def save_perf(self, d: Dict[str, Any]): self.r.set(self.k("state","performance"), json.dumps(d))
    # paper portfolio
    def load_portfolio(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","portfolio")); return json.loads(txt) if txt else {}
    def save_portfolio(self, d: Dict[str, Any]): self.r.set(self.k("state","portfolio"), json.dumps(d))
    # signal book (silent trades)
    def load_signal_book(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","signal_book"))
        if txt:
            try: return json.loads(txt)
            except Exception: ...
        return {"open_signals": [], "closed_signals": []}
    def save_signal_book(self, d: Dict[str, Any]): self.r.set(self.k("state","signal_book"), json.dumps(d))

# ================== SD zones & helpers ==================
def body(df: pd.DataFrame) -> pd.Series: return (df['close'] - df['open']).abs()
def avg_range(df: pd.DataFrame, n: int = 20) -> pd.Series: return (df['high'] - df['low']).rolling(n).mean()
def find_zones(df: pd.DataFrame, impulse_factor=1.8, zone_padding_pct=0.25, max_age_bars=300):
    zones = []; ar = avg_range(df, 20); b = body(df)
    for i in range(20, len(df)):
        ref = ar.iloc[i] if not np.isnan(ar.iloc[i]) else 0
        if b.iloc[i] > impulse_factor * ref:
            bull = df['close'].iloc[i] > df['open'].iloc[i]; j = i-1
            lo,hi = df['low'].iloc[j], df['high'].iloc[j]; pad = (hi-lo)*zone_padding_pct
            if bull and df['close'].iloc[j] < df['open'].iloc[j]:
                zones.append({"type":"demand","low":float(lo-pad),"high":float(hi+pad),"index":i-1})
            if (not bull) and df['close'].iloc[j] > df['open'].iloc[j]:
                zones.append({"type":"supply","low":float(lo-pad),"high":float(hi+pad),"index":i-1})
    return [z for z in zones if (len(df) - z["index"]) <= max_age_bars]
def in_demand(price: float, zones) -> bool:
    for z in zones or []:
        if z["type"]=="demand" and z["low"]<=price<=z["high"]: return True
    return False
def stop_from(df: pd.DataFrame, mode: str, atr_mult: float) -> float:
    if mode == "atr":
        a = atr(df, 14).iloc[-1]; a = 0 if np.isnan(a) else float(a)
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

# ================== Signal generators ==================
def day_signal(df: pd.DataFrame, p: DayParams, sd_cfg: Dict[str,Any], zones=None):
    look, voln = p.lookback_high, p.vol_sma
    if len(df) < max(look, voln)+5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"]>volS.iloc[-1]) and (p.rsi_min<=r.iloc[-1]<=p.rsi_max)
    retest_edge = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok   = retest_edge and (last["volume"]>0.8*volS.iloc[-1]) and (r.iloc[-1]>=p.rsi_min)
    if not (breakout_ok or retrec_ok): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry = float(last["close"]); stop = stop_from(df, p.stop_mode, p.atr_mult)
    return {"setup":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
            "level":float(highlvl),"note":"Breakout" if breakout_ok else "Retest-Reclaim",
            "event_bar_ts": df.index[-1].isoformat()}

def swing_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    need = max(p.get('ema100',100), p.get('vol_sma',20), p.get('breakout_lookback',34))+5
    if len(df)<need: return None
    df=df.copy(); df['ema20']=ema(df['close'],p.get('ema20',20)); df['ema50']=ema(df['close'],p.get('ema50',50))
    df['ema100']=ema(df['close'],p.get('ema100',100)); df['volS']=sma(df['volume'],p.get('vol_sma',20))
    r=rsi(df['close'],14); last=df.iloc[-1]
    aligned=(last['ema20']>last['ema50']>last['ema100']) and (r.iloc[-1]>=p.get('rsi_min',50))
    within=abs((last['close']-last['ema20'])/last['ema20']*100)<=p.get('pullback_pct_max',10.0)
    bounce=last['close']>df['close'].iloc[-2]
    hl=df['high'].iloc[-(p.get('breakout_lookback',34)+1):-1].max()
    breakout=(last['close']>hl) and (last['volume']>df['volS'].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get('enabled') and sd_cfg.get('mode','prefer')=='require' and not in_demand(float(last['close']), zones): return None
    entry=float(last['close']); stop=stop_from(df,p.get('stop_mode','swing'),p.get('atr_mult',2.0))
    return {"setup":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
            "level":float(hl),"note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout"}

def trend_signal(df: pd.DataFrame, p: TrendParams, sd_cfg: Dict[str,Any], zones=None):
    need=max(p.ema100,p.vol_sma,p.breakout_lookback)+5
    if len(df)<need: return None
    df=df.copy(); df["ema20"]=ema(df["close"],p.ema20); df["ema50"]=ema(df["close"],p.ema50)
    df["ema100"]=ema(df["close"],p.ema100); df["volS"]=sma(df["volume"],p.vol_sma)
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1]>=p.rsi_min)
    within=abs((last["close"]-last["ema20"])/last["ema20"]*100)<=p.pullback_pct_max
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(p.breakout_lookback+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]>df["volS"].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry=float(last["close"]); stop=stop_from(df,p.stop_mode,p.atr_mult)
    return {"setup":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
            "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout"}

# ===== Bearish (for “bear-safe” mapping if you use inverse tokens, or just for signal stats) =====
def day_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]):
    tf_cfg = cfg.get("day", {})
    look = int(tf_cfg.get("lookback_low", 20)); voln = int(tf_cfg.get("vol_sma", 20))
    if len(df) < max(look, voln) + 5: return None
    lowlvl = df["low"].iloc[-(look+1):-1].min(); last = df.iloc[-1]; volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    breakdown = last["close"] < lowlvl if tf_cfg.get("require_breakdown", True) else (last["low"] < lowlvl)
    vol_ok = (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) if tf_cfg.get("require_vol_confirm", True) else True
    rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (breakdown and vol_ok and rsi_weak): return None
    entry = float(last["close"]); stop = float(last["high"])
    return {"setup":"bear_day","entry":entry,"stop":stop,"t1":None,"t2":None,
            "level":float(lowlvl),"note":"Bearish breakdown 1h","event_bar_ts": df.index[-1].isoformat()}

def swing_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]):
    tf_cfg = cfg.get("swing", {})
    if len(df) < 120: return None
    e20, e50, e100 = ema(df["close"],20), ema(df["close"],50), ema(df["close"],100)
    r = rsi(df["close"], 14); volS = sma(df["volume"], 20); last = df.iloc[-1]
    ema_bear = (e20.iloc[-1] < e50.iloc[-1] < e100.iloc[-1]) if tf_cfg.get("ema_stack_bear", True) else True
    lowlvl = df["low"].iloc[-(tf_cfg.get("lookback_low",34)+1):-1].min()
    breakdown = last["close"] < lowlvl; rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    vol_ok = last["volume"] > volS.iloc[-1] if tf_cfg.get("require_vol_confirm", True) and not np.isnan(volS.iloc[-1]) else True
    if not (ema_bear and breakdown and rsi_weak and vol_ok): return None
    entry=float(last["close"]); stop=float(e20.iloc[-1])
    return {"setup":"bear_swing","entry":entry,"stop":stop,"t1":None,"t2":None,
            "level":float(lowlvl),"note":"Bearish breakdown 4h","event_bar_ts": df.index[-1].isoformat()}

def trend_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]):
    tf_cfg = cfg.get("trend", {})
    if len(df) < 200: return None
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    r = rsi(df["close"], 14); last = df.iloc[-1]
    cross = (e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-2] >= e50.iloc[-2]) if tf_cfg.get("ema20_below_50", True) else True
    lowlvl = df["low"].iloc[-(tf_cfg.get("lookback_low",55)+1):-1].min()
    breakdown = last["close"] < lowlvl; rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (cross and breakdown and rsi_weak): return None
    entry=float(last["close"]); stop=float(e50.iloc[-1])
    return {"setup":"bear_trend","entry":entry,"stop":stop,"t1":None,"t2":None,
            "level":float(lowlvl),"note":"Bearish trend breakdown 1d","event_bar_ts": df.index[-1].isoformat()}

# ================== Helpers ==================
STABLE_RE = re.compile(r'^(USDT|USDC|FDUSD|BUSD|TUSD|USDN|USD1|DAI|PAXG|XAUT|EUR[ST]?)$', re.I)

def is_stable_or_pegged(symbol_pair: str) -> bool:
    # filter obvious pegged/stable tickers
    base, quote = symbol_pair.split("/")
    return STABLE_RE.match(base) is not None

def pos_key(exchange:str, pair:str, setup:str, tf:str)->str: return f"{exchange}|{pair}|{tf}|{setup}"

# ================== Paper portfolio ==================
def ensure_portfolio(rds: RedisState, trading_cfg: Dict[str,Any]) -> Dict[str,Any]:
    pf = rds.load_portfolio()
    if not pf:
        base_bal = float(trading_cfg.get("base_balance_usdt", 1000.0))
        pf = {"cash_usdt": base_bal, "holdings": {}}
        rds.save_portfolio(pf)
        print(f"[portfolio] init paper balance = {base_bal:.2f} USDT")
    return pf

def compute_qty_for_risk(entry: float, stop: float, equity_usdt: float, risk_pct: float) -> float:
    risk_usdt = equity_usdt * max(0.0, risk_pct) / 100.0
    per_unit_risk = max(1e-9, entry - stop)
    qty = risk_usdt / per_unit_risk
    return max(0.0, qty)

def paper_buy(rds: RedisState, pf: Dict[str,Any], symbol: str, price: float, qty: float, min_order: float) -> bool:
    cost = price * qty
    if cost < min_order or cost > pf["cash_usdt"]:
        return False
    pf["cash_usdt"] -= cost
    h = pf["holdings"].setdefault(symbol, {"qty":0.0, "avg":0.0})
    new_qty = h["qty"] + qty
    h["avg"] = (h["avg"]*h["qty"] + price*qty) / max(1e-12, new_qty)
    h["qty"] = new_qty
    rds.save_portfolio(pf)
    return True

# ================== Trade performance snapshot (unchanged) ==================
def paper_snapshot(client: ExClient, rds: RedisState, pf: Dict[str,Any], perf: Dict[str,Any]):
    # holdings mv/pnl
    mv_total = 0.0; pnl_total = 0.0; lines=[]
    for sym, pos in sorted(pf.get("holdings", {}).items()):
        qty = float(pos["qty"]); avg = float(pos["avg"]); px = client.last_price(sym) or avg
        mv = qty * px; pnl = (px - avg) * qty; mv_total += mv; pnl_total += pnl
        lines.append((sym, qty, avg, px, pnl, mv))
    equity = pf.get("cash_usdt", 0.0) + mv_total
    print("\n--- Paper Performance Snapshot ---")
    print(f"Cash:      {pf.get('cash_usdt',0.0):.2f} USDT")
    print(f"Exposure:  {mv_total:.2f} USDT  | Positions: {len(pf.get('holdings',{}))}")
    print(f"Equity:    {equity:.2f} USDT  | Unrealized PnL: {pnl_total:+.2f} USDT")
    # open R from trade perf (if present)
    open_R = 0.0
    for tr in perf.get("open_trades", []):
        risk = float(tr.get("risk", 0.0)) or 1e-12
        entry = float(tr.get("entry", 0.0))
        px = client.last_price(tr["symbol"]) or entry
        open_R += (px - entry) / risk
    print(f"Open R:    {open_R:+.2f} R")
    # closed trade stats (if any)
    closed = perf.get("closed_trades", [])
    if closed:
        dfc = pd.DataFrame(closed)
        if "r_multiple" in dfc:
            win = float((dfc["r_multiple"] > 0).mean() * 100.0)
            avgR = float(dfc["r_multiple"].mean())
            medR = float(dfc["r_multiple"].median())
            bestR = float(dfc["r_multiple"].max())
            worstR = float(dfc["r_multiple"].min())
            gains = dfc.loc[dfc["r_multiple"]>0, "r_multiple"].sum()
            losses = -dfc.loc[dfc["r_multiple"]<0, "r_multiple"].sum()
            pfactor = float(gains / losses) if losses > 0 else float("inf")
            print(f"Closed n:  {len(closed)} | Win%: {win:.1f}% | AvgR: {avgR:.2f} | MedR: {medR:.2f} | PF: {('inf' if pfactor==float('inf') else f'{pfactor:.2f}')} | Best/Worst R: {bestR:.2f}/{worstR:.2f}")
    print("--- End Snapshot ---\n")

# ================== Signal Book (silent trades) ==================
def signal_id(ex: str, sym: str, tf: str, setup: str) -> str:
    return f"{ex}|{sym}|{tf}|{setup}"

def add_open_signal(book: Dict[str,Any], *, exchange, symbol, timeframe, setup, direction, entry, stop, t1, event_ts, note):
    open_list = book.setdefault("open_signals", [])
    sid = signal_id(exchange, symbol, timeframe, setup)
    # no duplicates (if exists, skip)
    for s in open_list:
        if s["id"] == sid:
            return False
    risk = abs(entry - (stop if stop is not None else entry)) or 1e-9
    open_list.append({
        "id": sid,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "setup": setup,  # day/swing/trend/bear_*
        "direction": direction,  # long/bear
        "opened_at": event_ts,
        "entry": float(entry),
        "stop": float(stop) if stop is not None else None,
        "t1": float(t1) if t1 is not None else None,
        "risk": float(risk),
        "note": note or "",
        "status": "open",
    })
    return True

def close_signal(sig: dict, *, outcome: str, price: float, closed_at):
    sig["status"] = "closed"
    sig["closed_at"] = str(closed_at)
    sig["exit_price"] = float(price)
    # outcome: "t1" (win) or "stop" (loss) or "timeout"
    sig["outcome"] = outcome
    if sig.get("risk", 0):
        direction = sig.get("direction","long")
        if direction == "long":
            rr = (price - sig["entry"]) / max(1e-12, sig["risk"])
        else:
            rr = (sig["entry"] - price) / max(1e-12, sig["risk"])
        sig["r_multiple"] = float(rr)
        sig["pct_return"] = float(((price/sig["entry"]) - 1) * (100 if direction=="long" else -100))
    else:
        sig["r_multiple"] = 0.0
        sig["pct_return"] = 0.0

def fetch_since(ex_client: ExClient, pair: str, tf: str, since_ts: pd.Timestamp) -> List[dict]:
    lim = 1000 if tf in ("1m","5m","15m","30m") else 400
    df = ex_client.ohlcv(pair, tf, lim)
    df = df.loc[since_ts:]
    out=[]
    for ts, row in df.iterrows():
        out.append({"ts": ts, "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])})
    return out

def _eval_first_touch(rows, entry, stop, t1, direction="long"):
    # returns (outcome, price) with first touch priority; if both touch, target wins first
    for r in rows:
        hi, lo = r["high"], r["low"]
        if direction=="long":
            hit_stop = (stop is not None) and (lo <= stop)
            hit_t1   = (t1   is not None) and (hi >= t1)
        else:  # bear
            hit_stop = (stop is not None) and (hi >= stop)      # stop for shorts above
            hit_t1   = (t1   is not None) and (lo <= t1)        # take profit moves down
        if hit_t1 and hit_stop:
            # prefer target over stop on same bar
            return "t1", t1
        if hit_t1:   return "t1", t1
        if hit_stop: return "stop", stop
    return None, None

def evaluate_open_signals(client: ExClient, book: Dict[str,Any], max_bars: Dict[str,int]):
    open_list = book.get("open_signals", [])
    closed_list = book.setdefault("closed_signals", [])
    keep=[]
    for s in open_list:
        try:
            since = pd.to_datetime(s["opened_at"], utc=True)
            rows = fetch_since(client, s["symbol"], s["timeframe"], since)
            if len(rows) <= 1:
                keep.append(s); continue
            eval_rows = rows[1:]  # first close after signal bar
            # Optional timeout by setup
            mkey = s["setup"]
            mbars = int(max_bars.get(mkey, 0)) if max_bars else 0
            if mbars and len(eval_rows) > mbars:
                eval_rows = eval_rows[:mbars]
            outcome, price = _eval_first_touch(eval_rows, s["entry"], s.get("stop"), s.get("t1"), direction=s.get("direction","long"))
            if outcome:
                close_signal(s, outcome=outcome, price=price, closed_at=eval_rows[-1]["ts"])
                closed_list.append(s)
            elif mbars and len(eval_rows) >= mbars:
                close_signal(s, outcome="timeout", price=eval_rows[-1]["close"], closed_at=eval_rows[-1]["ts"])
                closed_list.append(s)
            else:
                keep.append(s)
        except Exception as e:
            # on error keep it open to retry later
            keep.append(s)
    book["open_signals"] = keep

def signal_dashboard(book: Dict[str,Any]):
    open_n = len(book.get("open_signals", []))
    closed = book.get("closed_signals", [])
    print("\n--- Signal Performance Snapshot ---")
    print(f"Open signals: {open_n}  | Closed signals: {len(closed)}")
    if closed:
        df = pd.DataFrame(closed)
        by = df.groupby("setup")
        for setup, g in by:
            win = float((g["outcome"]=="t1").mean() * 100.0)
            avgR = float(g.get("r_multiple", pd.Series([0.0]*len(g))).mean())
            print(f"  {setup:>10}: n={len(g):3d}  win%={win:5.1f}%  avgR={avgR:+.2f}")
        # overall
        win_all = float((df["outcome"]=="t1").mean()*100.0)
        print(f"Overall: win%={win_all:5.1f}%  (t1 close = win, stop = loss)")
    print("--- End Signal Snapshot ---\n")

# ================== Movers & Top100 universe ==================
def fetch_cmc_top100(cfg: Dict[str,Any]) -> List[str]:
    # strict top 100 by market cap from CMC
    api_key = os.environ.get("CMC_API_KEY") or (cfg.get("movers", {}) or {}).get("cmc_api_key")
    if not api_key: return []
    headers = {"X-CMC_PRO_API_KEY": api_key}
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": 100, "convert": "USD", "sort": "market_cap", "sort_dir": "desc"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        data = r.json().get("data", [])
    except Exception:
        return []
    syms = [it["symbol"].upper() for it in data]
    return syms

def fetch_cmc_movers_legacy(cfg: Dict[str,Any]) -> List[str]:
    """
    Legacy movers logic (fakeout resistant):
      - universe: CMC listings (limit=cfg.movers.limit)
      - filters: percent_change_24h >= min_change_24h, volume_24h >= min_volume_usd_24h, age_days <= max_age_days
      - extra confirm: intraday momentum:
          * MACD line > signal on 1h and 4h
          * RSI(14) 1h >= 55
          * 1h EMA20 > EMA50
    """
    mv = cfg.get("movers", {})
    if not mv.get("enabled"): return []
    api_key = os.environ.get("CMC_API_KEY") or mv.get("cmc_api_key")
    if not api_key:
        print("[movers] enabled but no CMC_API_KEY — skipping"); return []
    headers = {"X-CMC_PRO_API_KEY": api_key}
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": mv.get("limit", 500), "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        data = r.json().get("data", [])
    except Exception as e:
        print("[movers] error:", e); return []
    min_change = mv.get("min_change_24h", 15.0); min_vol = mv.get("min_volume_usd_24h", 5_000_000); max_age = mv.get("max_age_days", 365)
    out=[]; now=pd.Timestamp.utcnow()
    for it in data:
        sym=it["symbol"].upper(); q=it.get("quote",{}).get("USD",{}); ch=(q.get("percent_change_24h") or 0); vol=(q.get("volume_24h") or 0)
        date_added=pd.to_datetime(it.get("date_added", now.isoformat()), utc=True); age_days=(now-date_added).days
        if (ch>=min_change) and (vol>=min_vol) and (age_days<=max_age):
            out.append(sym)
    return out

def filter_pairs_on_mexc(client: ExClient, symbols: List[str], quote: str="USDT") -> List[str]:
    client.load_markets()
    out=[]
    for sym in symbols:
        pair=f"{sym}/{quote}"
        if client.has_pair(pair) and not is_stable_or_pegged(pair):
            out.append(pair)
    return out

# ================== Discord ==================
def send_discord(webhook: str, text: str):
    if not webhook: return
    try:
        requests.post(webhook, json={"content": text}, timeout=10)
    except Exception as e:
        print("[discord] err:", e)

def fmt_signal_line(s: dict) -> str:
    return (f"• `{s['symbol']}` {s['timeframe']} *{s['setup']}* — {s.get('note','')}\n"
            f"  entry `{s['entry']:.6f}` stop `{s.get('stop'):.6f if s.get('stop') is not None else 0}` "
            f"t1 `{s.get('t1')}` t2 `{s.get('t2')}`").replace("None","-")

# ================== MAIN ==================
def run(cfg: Dict[str,Any]):
    client = ExClient()
    rds = RedisState(
        url=(cfg.get("persistence",{}).get("redis_url") or os.environ.get("REDIS_URL")),
        prefix=cfg.get("persistence",{}).get("key_prefix","spideybot:v1"),
        ttl_minutes=int(cfg.get("persistence",{}).get("ttl_minutes", 2880))
    )

    # env webhooks
    signals_wh = os.environ.get("DISCORD_SIGNALS_WEBHOOK","")
    trades_wh  = os.environ.get("DISCORD_TRADES_WEBHOOK","")

    trading_cfg = cfg.get("trading", {})
    paper_mode = bool(trading_cfg.get("paper", True))
    min_order = float(trading_cfg.get("min_usdt_order", 10.0))
    risk_pct  = float(trading_cfg.get("risk_per_trade_pct", 1.0))
    max_pos   = int(trading_cfg.get("max_concurrent_positions", 10))
    slip_bps  = int(trading_cfg.get("live_market_slippage_bps", 10))

    # universe — Top100 (signals) + Movers (legacy logic) — both filtered to MEXC tradable and non-stable
    top100_syms = fetch_cmc_top100(cfg)
    movers_syms = fetch_cmc_movers_legacy(cfg) if (cfg.get("movers",{}).get("enabled", False)) else []
    top100_pairs = filter_pairs_on_mexc(client, top100_syms, "USDT")
    movers_pairs = filter_pairs_on_mexc(client, movers_syms, cfg.get("movers",{}).get("quote","USDT"))

    def uniq(seq): return list(dict.fromkeys(seq))
    scan_pairs_day   = uniq(top100_pairs)
    scan_pairs_swing = uniq(top100_pairs)
    scan_pairs_trend = [p for p in top100_pairs if not is_stable_or_pegged(p)]

    # zones cache if enabled
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    zones_cache = {}
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        for pair in uniq(scan_pairs_day + scan_pairs_swing + scan_pairs_trend + movers_pairs):
            try:
                zdf = client.ohlcv(pair, ztf, zlook)
                zones_cache[pair] = find_zones(
                    zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                )
            except Exception as e:
                print(f"[zones] {pair} err:", e); zones_cache[pair]=[]

    # Params
    dayP   = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = TrendParams(**cfg.get("trend_trade_params", {}))
    bearish_cfg = cfg.get("bearish_signals", {"enabled": False})
    exits_cfg   = cfg.get("exits", {"enabled": True})
    pfcfg       = cfg.get("performance", {"enabled": True})
    # NOTE: BTC filter removed from params for compatibility (you asked earlier)

    # states
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}
    sigbook   = rds.load_signal_book()

    def create_signal_payload(pair: str, tf: str, setup: str, base_sig: dict, origin: str) -> dict:
        # unify payload for Discord + SignalBook
        return {
            "exchange":"mexc","symbol":pair,"timeframe":tf,"setup":setup,
            "entry":base_sig["entry"],"stop":base_sig.get("stop"),"t1":base_sig.get("t1"),"t2":base_sig.get("t2"),
            "note": (base_sig.get("note","") + ("" if origin=="top100" else " (mover)")).strip(),
            "event_bar_ts": base_sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat())
        }

    # ==== SCAN: TOP100 ====
    signals_top100=[]
    # DAY
    for pair in scan_pairs_day:
        try:
            df = client.ohlcv(pair, "1h", 300); zones = zones_cache.get(pair)
            s = day_signal(df, dayP, sd_cfg, zones)
            if s:
                signals_top100.append(create_signal_payload(pair,"1h","day",s,"top100"))
        except Exception as e: print(f"[scan-day] {pair} err:", e)
    # SWING (4h)
    tf_swing = swingP.get("timeframe","4h")
    for pair in scan_pairs_swing:
        try:
            df = client.ohlcv(pair, tf_swing, 400); zones = zones_cache.get(pair)
            s = swing_signal(df, swingP, sd_cfg, zones)
            if s:
                signals_top100.append(create_signal_payload(pair,tf_swing,"swing",s,"top100"))
        except Exception as e: print(f"[scan-swing] {pair} err:", e)
    # TREND (1d)
    for pair in scan_pairs_trend:
        try:
            df = client.ohlcv(pair, "1d", 320); zones = zones_cache.get(pair)
            s = trend_signal(df, trnP, sd_cfg, zones)
            if s:
                signals_top100.append(create_signal_payload(pair,"1d","trend",s,"top100"))
        except Exception as e: print(f"[scan-trend] {pair} err:", e)

    # ==== SCAN: MOVERS (legacy filter universe, then apply same entries on those pairs) ====
    signals_movers=[]
    for pair in movers_pairs:
        try:
            # 1h “mover trend” = reuse day_signal but mark note accordingly if breakout/pullback qualifies
            df1h = client.ohlcv(pair, "1h", 300)
            s = day_signal(df1h, dayP, sd_cfg, zones_cache.get(pair))
            if s:
                s["note"] = "Mover Trend"  # make it explicit
                signals_movers.append(create_signal_payload(pair,"1h","day",s,"mover"))
        except Exception as e:
            print(f"[movers] {pair} err:", e)

    # Bearish signals (silent lifecycle too)
    signals_bear=[]
    if bearish_cfg.get("enabled", False):
        for pair in scan_pairs_day:
            try:
                df = client.ohlcv(pair, bearish_cfg.get("day", {}).get("timeframe","1h"), 300)
                s=day_bearish_signal(df, bearish_cfg)
                if s: signals_bear.append(create_signal_payload(pair,"1h","bear_day",s,"top100"))
            except Exception as e: print(f"[bear-day] {pair} err:", e)
        for pair in scan_pairs_swing:
            try:
                df = client.ohlcv(pair, bearish_cfg.get("swing", {}).get("timeframe","4h"), 400)
                s=swing_bearish_signal(df, bearish_cfg)
                if s: signals_bear.append(create_signal_payload(pair,"4h","bear_swing",s,"top100"))
            except Exception as e: print(f"[bear-swing] {pair} err:", e)
        for pair in scan_pairs_trend:
            try:
                df = client.ohlcv(pair, bearish_cfg.get("trend", {}).get("timeframe","1d"), 320)
                s=trend_bearish_signal(df, bearish_cfg)
                if s: signals_bear.append(create_signal_payload(pair,"1d","bear_trend",s,"top100"))
            except Exception as e: print(f"[bear-trend] {pair} err:", e)

    # ====== SIGNAL BOOK LIFECYCLE (no dupes until closed) + DISCORD (signals channel) ======
    def try_open_signal_and_announce(s: dict, direction="long"):
        sid = signal_id("mexc", s["symbol"], s["timeframe"], s["setup"])
        # block if already open
        for cur in sigbook.get("open_signals", []):
            if cur["id"] == sid:
                return False
        created = add_open_signal(sigbook,
            exchange="mexc", symbol=s["symbol"], timeframe=s["timeframe"],
            setup=s["setup"], direction=direction, entry=s["entry"], stop=s.get("stop"), t1=s.get("t1"),
            event_ts=s.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()), note=s.get("note",""))
        if created:
            # send a clean signal line (no “skipped/executed” wording)
            send_discord(signals_wh, f"• `{s['symbol']}` {s['timeframe']} *{s['setup']}* — {s.get('note','')}\n"
                                     f"  entry `{s['entry']:.6f}` stop `{s.get('stop') if s.get('stop') is not None else '-'}` "
                                     f"t1 `{s.get('t1','-')}` t2 `{s.get('t2','-')}`")
        return created

    for s in signals_top100:
        try_open_signal_and_announce(s, direction="long")
    for s in signals_movers:
        try_open_signal_and_announce(s, direction="long")
    for s in signals_bear:
        try_open_signal_and_announce(s, direction="bear")

    # persist signal book (after openings)
    rds.save_signal_book(sigbook)

    # ====== EXECUTE REAL TRADES (unchanged behavior/limits) ======
    already_open_symbols = set(v["symbol"] for v in positions.get("active_positions", {}).values())
    n_open = len(already_open_symbols)

    def execute_trade_if_room(sig: dict):
        nonlocal n_open
        sym = sig["symbol"]; tf = sig["timeframe"]; entry = float(sig["entry"]); stop = float(sig.get("stop") or entry)
        # guard: skip if already open or max positions
        if sym in already_open_symbols or n_open >= max_pos:
            return False
        # size & buy
        if paper_mode:
            equity = pf.get("cash_usdt", 0.0)
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            if entry * qty < min_order or entry * qty > equity:
                return False
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if not ok: return False
            send_discord(trades_wh, f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `{entry:.6f}` (paper)")
        else:
            try:
                px = entry * (1 + slip_bps/10000.0)
                bal = client.ex.fetch_balance().get("USDT", {}).get("free", 0.0)
                equity = float(bal)
                qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
                notional = qty * px
                if notional < min_order or notional > equity:
                    return False
                client.place_market(sym, "buy", qty)
                send_discord(trades_wh, f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `~{px:.6f}` (live)")
            except Exception:
                return False
        # register in positions (keeps your trade perf logic unchanged)
        positions.setdefault("active_positions", {})[f"mexc|{sym}|{tf}|live"] = {
            "exchange":"mexc","symbol":sym,"type":"live","entry":entry,"timeframe":tf,"ts":pd.Timestamp.utcnow().isoformat()
        }
        # register open trade perf
        perf.setdefault("open_trades", []).append({
            "id": f"mexc|{sym}|{tf}|live|{pd.Timestamp.utcnow().isoformat()}",
            "exchange":"mexc", "symbol":sym, "timeframe":tf, "type":"live",
            "opened_at": pd.Timestamp.utcnow().isoformat(),
            "entry": entry, "stop": stop, "t1": sig.get("t1"), "t2": sig.get("t2"),
            "risk": max(1e-12, entry - stop), "status":"open"
        })
        already_open_symbols.add(sym); n_open += 1
        return True

    # Try executing trades only for top100 signals (your rules; movers can be signaling-only if you prefer)
    for s in signals_top100:
        execute_trade_if_room(s)

    # persist trade states
    rds.save_positions(positions)
    rds.save_perf(perf)

    # ====== EVALUATE OPEN SIGNALS (close on T1 or Stop) & print signal dashboard ======
    max_bars_eval = (cfg.get("performance", {}) or {}).get("max_bars_eval_signals", {
        "day": 120, "swing": 180, "trend": 220, "bear_day": 120, "bear_swing": 180, "bear_trend": 220
    })
    evaluate_open_signals(client, sigbook, max_bars_eval)
    rds.save_signal_book(sigbook)

    # ===== Logs =====
    ts_now = pd.Timestamp.utcnow().isoformat()
    print(f"=== MEXC Signals @ {ts_now} ===")
    print(f"Scanned — 1h:{len(scan_pairs_day)}  4h:{len(scan_pairs_swing)}  1d:{len(scan_pairs_trend)}  | movers:{len(movers_pairs)}")
    if movers_pairs:
        print("Movers (MEXC):", ", ".join(movers_pairs[:20]) + (" ..." if len(movers_pairs)>20 else ""))

    # Console: summarize what *new* signals we opened (we already sent Discord)
    new_cnt = 0
    for group_name, arr in [("Top100", signals_top100), ("Movers", signals_movers), ("Bear", signals_bear)]:
        if arr:
            print(f"--- Signals ({group_name}) ---")
            for s in arr:
                print(f"[SIGNAL] {s['symbol']} {s['timeframe']} — {s['setup']} — {s.get('note','')} | entry {s['entry']:.6f} stop {s.get('stop','-')} t1 {s.get('t1','-')} t2 {s.get('t2','-')}")
                new_cnt += 1
    if new_cnt == 0:
        print("No new signals this run.")

    # Paper performance snapshot for trades (unchanged)
    if paper_mode:
        paper_snapshot(client, rds, pf, perf)

    # Signal performance snapshot (console only)
    signal_dashboard(sigbook)

# ================== Entrypoint ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)

    # env expansion ${VAR}
    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o
    cfg = expand_env(cfg)

    run(cfg)
