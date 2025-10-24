#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mexc_trader_bot.py — MEXC spot-only scanner/trader with:
- Bear-safe mode (inverse "S" tokens) for bearish setups
- Paper/live trading
- Redis persistence (positions, portfolio, performance)
- Dynamic Top-100 by market cap (CMC) as watchlist (optional)
- Discord notifications split into two channels:
    1) Signals channel = ONLY signal info (entries & exits), no 'skipped' or 'executed' text
    2) Trades channel  = ONLY actual trades (paper or live), never 'none'

Key config flags:
- use_top100_dynamic_watchlist: true  # replaces symbols_watchlist/movers with CMC top-100 filtered to MEXC/USDT
- discord:
    enabled: true
    signals_webhook: "..."
    trades_webhook: "..."
- trading.paper: true|false
- trading.auto_sell_on_exit (paper only default true; live default false)

Defensive config parsing:
- Dataclasses are constructed with safe key filtering, so extra keys (e.g. 'multi_tf') won't crash.
"""

import argparse, json, os, sys, yaml, requests, math
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List, Tuple, Set, Optional
import pandas as pd
import numpy as np
import ccxt
import redis
from datetime import datetime, timezone

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
    btc_filter: bool = True
    btc_symbol: str = "BTC/USDT"
    btc_ema: int = 20
    stop_mode: str = "swing"
    atr_mult: float = 1.5
    # NOTE: multi_tf or any extra keys from config are ignored safely.

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

# ============== Safe dataclass parsing ==============
def dataclass_from_cfg(dc_cls, d: Dict[str,Any]):
    if not isinstance(d, dict): return dc_cls()
    allowed = {f.name for f in fields(dc_cls)}
    clean = {k:v for k,v in d.items() if k in allowed}
    return dc_cls(**clean)

# ================== Redis persistence ==================
class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url:
            raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL env.")
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48*3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        # smoke test
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")

    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])

    # edge memories
    def get_mem(self, kind: str, key: str) -> str:
        return self.r.get(self.k("mem", kind, key)) or ""
    def set_mem(self, kind: str, key: str, bar_iso: str):
        self.r.setex(self.k("mem", kind, key), self.ttl_seconds, bar_iso)

    # positions / performance
    def load_positions(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","active_positions"))
        return json.loads(txt) if txt else {}
    def save_positions(self, d: Dict[str, Any]):
        self.r.set(self.k("state","active_positions"), json.dumps(d))

    def load_perf(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","performance"))
        return json.loads(txt) if txt else {"open_trades": [], "closed_trades": []}
    def save_perf(self, d: Dict[str, Any]):
        self.r.set(self.k("state","performance"), json.dumps(d))

    # portfolio (paper)
    def load_portfolio(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","portfolio"))
        return json.loads(txt) if txt else {}
    def save_portfolio(self, d: Dict[str, Any]):
        self.r.set(self.k("state","portfolio"), json.dumps(d))

# ================== SD zones / helpers ==================
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

# ================== Signals ==================
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
    return {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
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
    return {"type":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
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
    return {"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
            "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout"}

# Bearish detectors
def day_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    tf_cfg = cfg.get("day", {})
    look = int(tf_cfg.get("lookback_low", 20)); voln = int(tf_cfg.get("vol_sma", 20))
    if len(df) < max(look, voln) + 5: return None
    lowlvl = df["low"].iloc[-(look+1):-1].min(); last = df.iloc[-1]; volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    breakdown = last["close"] < lowlvl if tf_cfg.get("require_breakdown", True) else (last["low"] < lowlvl)
    vol_ok = (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) if tf_cfg.get("require_vol_confirm", True) else True
    rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (breakdown and vol_ok and rsi_weak): return None
    entry = float(last["close"]); stop = float(last["high"])  # conservative
    return {"type":"bear_day","entry":entry,"stop":stop,"t1":None,"t2":None,
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
    return {"type":"bear_swing","entry":entry,"stop":stop,"t1":None,"t2":None,
            "level":float(lowlvl),"note":"Bearish breakdown 4h"}

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
    return {"type":"bear_trend","entry":entry,"stop":stop,"t1":None,"t2":None,
            "level":float(lowlvl),"note":"Bearish trend breakdown 1d"}

# ================== BTC market filter ==================
def btc_ok(ex: ExClient, dayp: DayParams) -> bool:
    try:
        df = ex.ohlcv(dayp.btc_symbol, "1h", 120)
        return df["close"].iloc[-1] > ema(df["close"], dayp.btc_ema).iloc[-1]
    except Exception:
        return True

# ================== CMC Top-100 Watchlist ==================
def fetch_cmc_top100_symbols(api_key: str) -> List[str]:
    headers = {"X-CMC_PRO_API_KEY": api_key}
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": 100, "convert": "USD", "sort": "market_cap", "sort_dir": "desc"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    symbols = [it["symbol"].upper() for it in data]
    return symbols

def filter_pairs_on_mexc(client: ExClient, symbols: List[str], quote: str="USDT") -> List[str]:
    client.load_markets()
    out=[]
    for sym in symbols:
        pair=f"{sym}/{quote}"
        if client.has_pair(pair): out.append(pair)
    return out

# ================== Bear-safe mapping ==================
def map_bearish_to_inverse_token(client: ExClient, pair: str, suffixes: List[str]) -> Optional[str]:
    base, quote = pair.split("/")
    for suf in suffixes:
        alt = f"{base}{suf}/{quote}"
        if client.has_pair(alt):
            return alt
    return None

# ================== Performance + state helpers ==================
def pos_key(exchange:str, pair:str, sig_type:str)->str: return f"{exchange}|{pair}|{sig_type}"

def add_open_trade(perf: Dict[str,Any], *, exchange, symbol, tf, sig_type, entry, stop, t1, t2, event_ts):
    risk = max(1e-12, entry - stop)
    perf.setdefault("open_trades", []).append({
        "id": f"{exchange}|{symbol}|{tf}|{sig_type}|{event_ts}",
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": tf,
        "type": sig_type,
        "opened_at": event_ts,
        "entry": float(entry),
        "stop": float(stop),
        "t1": float(t1) if t1 else None,
        "t2": float(t2) if t2 else None,
        "risk": float(risk),
        "status": "open",
    })

def update_positions_active(positions: Dict[str,Any], signal: Dict[str,Any]):
    ap = positions.setdefault("active_positions",{})
    k = pos_key(signal["exchange"], signal["symbol"], signal["type"])
    ap[k] = {"exchange":signal["exchange"],"symbol":signal["symbol"],"type":signal["type"],
             "entry":signal["entry"],"timeframe":signal["timeframe"],"ts":pd.Timestamp.utcnow().isoformat()}

def remove_position(positions: Dict[str,Any], exchange:str, pair:str, sig_type:str):
    ap = positions.setdefault("active_positions",{})
    ap.pop(pos_key(exchange,pair,sig_type), None)

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

def paper_sell_all(rds: RedisState, pf: Dict[str,Any], symbol: str, price: float) -> float:
    h = pf["holdings"].get(symbol)
    if not h: return 0.0
    proceeds = price * h["qty"]
    pf["cash_usdt"] += proceeds
    pf["holdings"].pop(symbol, None)
    rds.save_portfolio(pf)
    return proceeds

# ================== Discord ==================
def _discord_post(webhook: str, content: str):
    if not webhook or not content.strip(): return
    try:
        requests.post(webhook, json={"content": content}, timeout=10)
    except Exception as e:
        print("[discord] err:", e)

def fmt_signal_line(s: dict) -> str:
    t1 = s.get('t1'); t2 = s.get('t2')
    t1txt = f"{t1:.6f}" if t1 else "-"
    t2txt = f"{t2:.6f}" if t2 else "-"
    return (f"• `{s['symbol']}` {s['timeframe']} {s['type'].upper()} — {s.get('note','')}\n"
            f"  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{t1txt}` t2 `{t2txt}`")

def fmt_exit_line(x: dict) -> str:
    price = x.get("price")
    ptxt = f" @ `{price:.6f}`" if price is not None else ""
    return f"• `{x['symbol']}` {x['timeframe']} — {x['type']} — {x.get('reason','')}{ptxt}"

def fmt_trade_line(ev: dict) -> str:
    # ev: {side, symbol, qty, price, mode, timeframe}
    return (f"• {ev['side'].upper()} `{ev['symbol']}` {ev.get('timeframe','')} "
            f"qty `{ev['qty']:.6f}` @ `{ev['price']:.6f}` ({ev['mode']})")

# ================== Paper performance snapshot ==================
def paper_snapshot(client: ExClient, rds: RedisState, pf: Dict[str,Any], perf: Dict[str,Any]):
    mv_total = 0.0
    pnl_total = 0.0
    lines = []
    for sym, pos in sorted(pf.get("holdings", {}).items()):
        qty = float(pos["qty"]); avg = float(pos["avg"])
        px = client.last_price(sym) or avg
        mv = qty * px
        pnl = (px - avg) * qty
        mv_total += mv
        pnl_total += pnl
        lines.append((sym, qty, avg, px, pnl, mv))
    equity = pf.get("cash_usdt", 0.0) + mv_total
    exposure = mv_total

    open_R = 0.0
    open_details = []
    for tr in perf.get("open_trades", []):
        sym = tr.get("symbol")
        risk = float(tr.get("risk", 0.0)) or 1e-12
        entry = float(tr.get("entry", 0.0))
        px = client.last_price(sym) or entry
        r_now = (px - entry) / risk
        open_R += r_now
        open_details.append((sym, tr.get("timeframe",""), r_now, ((px/entry)-1.0)*100.0, entry, px))

    closed = perf.get("closed_trades", [])
    win = 0.0; avgR = 0.0; medR = 0.0; bestR = 0.0; worstR = 0.0; pfactor = 0.0
    if closed:
        dfc = pd.DataFrame(closed)
        if "r_multiple" in dfc:
            win = float((dfc["r_multiple"] > 0).mean() * 100.0)
            avgR = float(dfc["r_multiple"].mean()); medR=float(dfc["r_multiple"].median())
            bestR=float(dfc["r_multiple"].max()); worstR=float(dfc["r_multiple"].min())
            gains = dfc.loc[dfc["r_multiple"]>0, "r_multiple"].sum()
            losses = -dfc.loc[dfc["r_multiple"]<0, "r_multiple"].sum()
            pfactor = float(gains / losses) if losses > 0 else float("inf")

    print("\n--- Paper Performance Snapshot ---")
    print(f"Cash:      {pf.get('cash_usdt',0.0):.2f} USDT")
    print(f"Exposure:  {exposure:.2f} USDT  | Positions: {len(pf.get('holdings',{}))}")
    print(f"Equity:    {equity:.2f} USDT  | Unrealized PnL: {pnl_total:+.2f} USDT")
    print(f"Open R:    {open_R:+.2f} R (sum across open trades)")
    if closed:
        print(f"Closed n:  {len(closed)} | Win%: {win:.1f}% | AvgR: {avgR:.2f} | MedR: {medR:.2f} | PF: {pfactor if pfactor!=float('inf') else 'inf'} | Best/Worst R: {bestR:.2f}/{worstR:.2f}")
    if lines:
        lines.sort(key=lambda x: x[4], reverse=True)
        top = lines[:5]
        print("Top PnL positions:")
        for sym, qty, avg, last, pnl, mv in top:
            pct = ((last/avg)-1.0)*100.0 if avg>0 else 0.0
            print(f"  {sym}: qty={qty:.6f} avg={avg:.6f} last={last:.6f} | PnL={pnl:+.2f} USDT ({pct:+.2f}%)")
    if open_details:
        open_details.sort(key=lambda x: abs(x[2]), reverse=True)
        topR = open_details[:5]
        print("Top |R| open trades:")
        for sym, tf, rnow, pct, entry, last in topR:
            print(f"  {sym} {tf}: R={rnow:+.2f} | {pct:+.2f}% | entry={entry:.6f} last={last:.6f}")
    print("--- End Snapshot ---\n")

# ================== MAIN ==================
def run(cfg: Dict[str,Any]):
    client = ExClient()
    rds = RedisState(
        url=(cfg.get("persistence",{}).get("redis_url") or os.environ.get("REDIS_URL")),
        prefix=cfg.get("persistence",{}).get("key_prefix","spideybot:v1"),
        ttl_minutes=int(cfg.get("persistence",{}).get("ttl_minutes", 2880))
    )

    trading_cfg = cfg.get("trading", {})
    paper_mode = bool(trading_cfg.get("paper", True))
    min_order = float(trading_cfg.get("min_usdt_order", 10.0))
    risk_pct  = float(trading_cfg.get("risk_per_trade_pct", 1.0))
    max_pos   = int(trading_cfg.get("max_concurrent_positions", 10))
    slip_bps  = int(trading_cfg.get("live_market_slippage_bps", 10))
    auto_sell_paper = bool(trading_cfg.get("auto_sell_on_exit", True))
    auto_sell_live  = bool(trading_cfg.get("auto_sell_on_exit_live", False))

    # Discord config
    dcfg = cfg.get("discord", {"enabled": False})
    discord_enabled = bool(dcfg.get("enabled", False))
    sig_hook = dcfg.get("signals_webhook")
    trd_hook = dcfg.get("trades_webhook")

    # Build universe
    watchlist: List[str] = []
    movers_pairs: List[str] = []

    # Dynamic Top-100 by market cap
    if bool(cfg.get("use_top100_dynamic_watchlist", False)):
        api_key = os.environ.get("CMC_API_KEY") or (cfg.get("movers", {}) or {}).get("cmc_api_key")
        if not api_key:
            print("[watchlist] Top100 mode requested but no CMC_API_KEY found. Falling back to symbols_watchlist.")
            watchlist = cfg.get("symbols_watchlist", []) or []
        else:
            try:
                syms_top100 = fetch_cmc_top100_symbols(api_key)
                watchlist = filter_pairs_on_mexc(client, syms_top100, "USDT")
                print(f"[watchlist] Top100 mode active — {len(watchlist)} MEXC pairs")
            except Exception as e:
                print("[watchlist] Top100 fetch err:", e)
                watchlist = cfg.get("symbols_watchlist", []) or []
    else:
        # static list + optional movers (unchanged logic)
        env_wl = os.environ.get("MEXC_WATCHLIST","")
        if env_wl.strip():
            watchlist = [s.strip() for s in env_wl.split(",") if s.strip()]
        else:
            watchlist = cfg.get("symbols_watchlist", []) or []

        if cfg.get("movers", {}).get("enabled", False):
            try:
                # reuse previous movers (500 gainers) filtered to MEXC
                headers = {"X-CMC_PRO_API_KEY": os.environ.get("CMC_API_KEY") or cfg["movers"].get("cmc_api_key")}
                if headers["X-CMC_PRO_API_KEY"]:
                    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
                    params = {"limit": cfg["movers"].get("limit", 500), "convert": "USD"}
                    r = requests.get(url, headers=headers, params=params, timeout=20)
                    data = r.json().get("data", [])
                    min_change = cfg["movers"].get("min_change_24h", 15.0)
                    min_vol = cfg["movers"].get("min_volume_usd_24h", 5_000_000)
                    now=pd.Timestamp.utcnow()
                    raw = [d for d in data if (d.get("quote",{}).get("USD",{}).get("percent_change_24h",0)>=min_change and
                                               d.get("quote",{}).get("USD",{}).get("volume_24h",0)>=min_vol)]
                    movers_syms = [d["symbol"].upper() for d in raw]
                    movers_pairs = filter_pairs_on_mexc(client, movers_syms, cfg["movers"].get("quote","USDT"))
            except Exception as e:
                print("[movers] err:", e)

    # Final scan sets
    def uniq(seq): return list(dict.fromkeys(seq))
    base_pairs = [p for p in watchlist if client.has_pair(p)]
    scan_pairs_day   = uniq(base_pairs + movers_pairs)
    scan_pairs_swing = uniq(base_pairs + movers_pairs)
    scan_pairs_trend = base_pairs

    # SD zones cache
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    zones_cache = {}
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        for pair in uniq(scan_pairs_day + scan_pairs_swing + scan_pairs_trend):
            try:
                zdf = client.ohlcv(pair, ztf, zlook)
                zones_cache[pair] = find_zones(
                    zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                )
            except Exception as e:
                print(f"[zones] {pair} err:", e); zones_cache[pair]=[]

    # Params (safe)
    dayP   = dataclass_from_cfg(DayParams, cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = dataclass_from_cfg(TrendParams, cfg.get("trend_trade_params", {}))
    bearish_cfg = cfg.get("bearish_signals", {"enabled": False})
    exits_cfg   = cfg.get("exits", {"enabled": True})
    pfcfg       = cfg.get("performance", {"enabled": True})
    bear_mode   = cfg.get("bear_mode", {"enabled": True})
    inv_suffix  = bear_mode.get("inverse_suffixes", ["3S","5S"])

    # Load states
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}

    results = {"ts": pd.Timestamp.utcnow().isoformat(), "signals": [], "exit_signals": [], "movers": {"mexc": movers_pairs}}

    # ===== entries (bullish) =====
    allow_day = (not dayP.btc_filter) or btc_ok(client, dayP)
    if dayP.btc_filter and not allow_day:
        print("[filter] BTC below EMA — day signals paused")

    # DAY (1h)
    for pair in scan_pairs_day:
        try:
            df1h = client.ohlcv(pair, "1h", 300)
            zones = zones_cache.get(pair)
            sig = day_signal(df1h, dayP, sd_cfg, zones)
            if sig and allow_day:
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc"})
                results["signals"].append(sig)
        except Exception as e:
            print(f"[scan-day] {pair} err:", e)

    # SWING (4h)
    tf_swing = swingP.get("timeframe","4h")
    for pair in scan_pairs_swing:
        try:
            df4h = client.ohlcv(pair, tf_swing, 400)
            zones = zones_cache.get(pair)
            sig = swing_signal(df4h, swingP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":tf_swing,"exchange":"mexc"})
                results["signals"].append(sig)
        except Exception as e:
            print(f"[scan-swing] {pair} err:", e)

    # TREND (1d)
    for pair in scan_pairs_trend:
        try:
            dfd = client.ohlcv(pair, "1d", 320)
            zones = zones_cache.get(pair)
            sig = trend_signal(dfd, trnP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1d","exchange":"mexc"})
                results["signals"].append(sig)
        except Exception as e:
            print(f"[scan-trend] {pair} err:", e)

    # ===== bearish setups -> inverse longs if available =====
    if bearish_cfg.get("enabled", True) and bear_mode.get("enabled", True):
        # DAY bearish
        for pair in scan_pairs_day:
            try:
                df1h = client.ohlcv(pair, bearish_cfg.get("day", {}).get("timeframe","1h"), 300)
                zones = zones_cache.get(pair)
                sig = day_bearish_signal(df1h, bearish_cfg, sd_cfg, zones)
                if sig:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        sig.update({"symbol":inv,"timeframe":"1h","exchange":"mexc","type":"inverse","note":f"Inverse LONG from bearish {pair}"})
                        results["signals"].append(sig)
                    else:
                        print(f"[bear-safe] No inverse token for {pair}")
            except Exception as e:
                print(f"[bear-day] {pair} err:", e)
        # SWING bearish
        for pair in scan_pairs_swing:
            try:
                df4h = client.ohlcv(pair, bearish_cfg.get("swing", {}).get("timeframe","4h"), 400)
                sig = swing_bearish_signal(df4h, bearish_cfg)
                if sig:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        sig.update({"symbol":inv,"timeframe":"4h","exchange":"mexc","type":"inverse","note":f"Inverse LONG from bearish {pair}"})
                        results["signals"].append(sig)
                    else:
                        print(f"[bear-safe] No inverse token for {pair}")
            except Exception as e:
                print(f"[bear-swing] {pair} err:", e)
        # TREND bearish
        for pair in scan_pairs_trend:
            try:
                dfd = client.ohlcv(pair, bearish_cfg.get("trend", {}).get("timeframe","1d"), 320)
                sig = trend_bearish_signal(dfd, bearish_cfg)
                if sig:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        sig.update({"symbol":inv,"timeframe":"1d","exchange":"mexc","type":"inverse","note":f"Inverse LONG from bearish {pair}"})
                        results["signals"].append(sig)
                    else:
                        print(f"[bear-safe] No inverse token for {pair}")
            except Exception as e:
                print(f"[bear-trend] {pair} err:", e)

    # ===== EXECUTE entries (paper or live) =====
    already_open_symbols = {v["symbol"] for v in positions.get("active_positions", {}).values()}
    n_open = len(already_open_symbols)

    # capture trade events (for trades channel)
    trade_events: List[Dict[str,Any]] = []

    for sig in results["signals"]:
        sym = sig["symbol"]; tf = sig["timeframe"]; typ = sig["type"]; entry = sig["entry"]; stop = sig["stop"]

        # skip execution if already open or max positions reached
        if (sym in already_open_symbols) or (n_open >= max_pos):
            continue

        # sizing/execution
        if paper_mode:
            equity = pf.get("cash_usdt", 0.0)
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if not ok:
                continue
            trade_events.append({"side":"buy","symbol":sym,"qty":qty,"price":entry,"mode":"paper","timeframe":tf})
            print(f"[paper] BUY {sym} {qty:.6f} @ {entry:.6f}")
        else:
            try:
                px = entry * (1 + slip_bps/10000.0)
                bal = client.ex.fetch_balance().get("USDT", {}).get("free", 0.0)
                equity = float(bal)
                qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
                notional = qty * px
                if notional < min_order or notional > equity:
                    continue
                client.place_market(sym, "buy", qty)
                trade_events.append({"side":"buy","symbol":sym,"qty":qty,"price":px,"mode":"live","timeframe":tf})
                print(f"[live] BUY {sym} {qty:.6f} @ ~{px:.6f}")
            except Exception as e:
                print(f"[live] order err {sym}:", e); continue

        # register state & perf
        signal_for_state = {"exchange":"mexc","symbol":sym,"type":typ if typ!="day" else "day","entry":entry,"timeframe":tf}
        update_positions_active(positions, signal_for_state)
        add_open_trade(perf, exchange="mexc", symbol=sym, tf=tf, sig_type=typ, entry=entry, stop=stop,
                       t1=sig.get("t1"), t2=sig.get("t2"), event_ts=sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()))
        already_open_symbols.add(sym)
        n_open += 1

    # ===== exits (edge-trigger) =====
    if exits_cfg.get("enabled", True):
        # collect active pairs regardless of watchlist
        active_pairs = sorted({v["symbol"] for v in positions.get("active_positions", {}).values()})
        check_pairs = sorted(set(base_pairs + active_pairs))
        for pair in check_pairs:
            try:
                # Day exit (1h)
                df1h = client.ohlcv(pair,"1h",200)
                # simple exit: price < EMA20 or RSI momentum rollover
                e20 = ema(df1h["close"], int(exits_cfg.get("day",{}).get("ema_break", 20)))
                r = rsi(df1h["close"], 14)
                exit_day = (df1h["close"].iloc[-1] < e20.iloc[-1]) and (r.iloc[-1] <= int(exits_cfg.get("day",{}).get("rsi_drop_to",60)))
                if exit_day:
                    px = float(df1h["close"].iloc[-1])
                    results["exit_signals"].append({"type":"day_exit","reason":"Close<EMA20 & RSI down","symbol":pair,"timeframe":"1h","exchange":"mexc","price":px})
                    # auto-sell if configured
                    if paper_mode and auto_sell_paper and pair in pf.get("holdings", {}):
                        proceeds = paper_sell_all(rds, pf, pair, px)
                        remove_position(positions, "mexc", pair, "day")
                        trade_events.append({"side":"sell","symbol":pair,"qty":0.0,"price":px,"mode":"paper","timeframe":"1h"})
                    elif (not paper_mode) and auto_sell_live:
                        try:
                            h = client.ex.fetch_balance().get(pair.split("/")[0], {})
                            qty = float(h.get("free", 0.0))
                            if qty > 0:
                                client.place_market(pair, "sell", qty)
                                remove_position(positions, "mexc", pair, "day")
                                trade_events.append({"side":"sell","symbol":pair,"qty":qty,"price":px,"mode":"live","timeframe":"1h"})
                        except Exception as e:
                            print(f"[live] sell err {pair}:", e)

                # Swing exit (4h)
                df4h = client.ohlcv(pair,"4h",300)
                e50 = ema(df4h["close"], int(exits_cfg.get("swing",{}).get("ema_break", 50)))
                r4 = rsi(df4h["close"], 14)
                exit_sw = (df4h["close"].iloc[-1] < e50.iloc[-1]) and (r4.iloc[-1] <= int(exits_cfg.get("swing",{}).get("rsi_below",50)))
                if exit_sw:
                    px = float(df4h["close"].iloc[-1])
                    results["exit_signals"].append({"type":"swing_exit","reason":"Close<EMA50 & RSI down","symbol":pair,"timeframe":"4h","exchange":"mexc","price":px})
                    if paper_mode and auto_sell_paper and pair in pf.get("holdings", {}):
                        proceeds = paper_sell_all(rds, pf, pair, px)
                        remove_position(positions, "mexc", pair, "swing")
                        trade_events.append({"side":"sell","symbol":pair,"qty":0.0,"price":px,"mode":"paper","timeframe":"4h"})
                    elif (not paper_mode) and auto_sell_live:
                        try:
                            h = client.ex.fetch_balance().get(pair.split("/")[0], {})
                            qty = float(h.get("free", 0.0))
                            if qty > 0:
                                client.place_market(pair, "sell", qty)
                                remove_position(positions, "mexc", pair, "swing")
                                trade_events.append({"side":"sell","symbol":pair,"qty":qty,"price":px,"mode":"live","timeframe":"4h"})
                        except Exception as e:
                            print(f"[live] sell err {pair}:", e)

                # Trend exit (1d)
                dfd = client.ohlcv(pair,"1d",320)
                e20d, e50d = ema(dfd["close"],20), ema(dfd["close"],50)
                rD = rsi(dfd["close"], 14)
                exit_tr = (e20d.iloc[-1] < e50d.iloc[-1]) and (rD.iloc[-1] <= int(exits_cfg.get("trend",{}).get("rsi_below",50)))
                if exit_tr:
                    px = float(dfd["close"].iloc[-1])
                    results["exit_signals"].append({"type":"trend_exit","reason":"EMA20<EMA50 & RSI down","symbol":pair,"timeframe":"1d","exchange":"mexc","price":px})
                    if paper_mode and auto_sell_paper and pair in pf.get("holdings", {}):
                        proceeds = paper_sell_all(rds, pf, pair, px)
                        remove_position(positions, "mexc", pair, "trend")
                        trade_events.append({"side":"sell","symbol":pair,"qty":0.0,"price":px,"mode":"paper","timeframe":"1d"})
                    elif (not paper_mode) and auto_sell_live:
                        try:
                            h = client.ex.fetch_balance().get(pair.split("/")[0], {})
                            qty = float(h.get("free", 0.0))
                            if qty > 0:
                                client.place_market(pair, "sell", qty)
                                remove_position(positions, "mexc", pair, "trend")
                                trade_events.append({"side":"sell","symbol":pair,"qty":qty,"price":px,"mode":"live","timeframe":"1d"})
                        except Exception as e:
                            print(f"[live] sell err {pair}:", e)

            except Exception as e:
                print(f"[exits] {pair} err:", e)

    # Save states
    rds.save_positions(positions)
    rds.save_perf(perf)

    # ===== Logs (console) =====
    print(f"=== MEXC Signals @ {results['ts']} ===")
    print(f"Scanned — 1h:{len(scan_pairs_day)}  4h:{len(scan_pairs_swing)}  1d:{len(scan_pairs_trend)}  | universe:{len(base_pairs)}  movers:{len(movers_pairs)}")
    if results["movers"]["mexc"]:
        print("Movers (MEXC):", ", ".join(results["movers"]["mexc"]))

    if results["signals"]:
        for s in results["signals"]:
            print(f"[SIGNAL] {s['symbol']} {s['timeframe']} — {s['type']} — {s['note']} | entry {s['entry']:.6f} stop {s['stop']:.6f} t1 {s.get('t1')} t2 {s.get('t2')}")
    else:
        print("No signals.")

    if results["exit_signals"]:
        for x in results["exit_signals"]:
            print(f"[EXIT] {x['symbol']} {x['timeframe']} — {x['type']} — {x['reason']} @ {x.get('price','-')}")
    else:
        print("No exit signals.")

    # Paper snapshot each run
    if paper_mode:
        paper_snapshot(client, rds, pf, perf)

    # ===== Discord notifications =====
    if discord_enabled:
        # Signals channel: ONLY the signals & exit signals (no meta/skip)
        sig_lines = []
        if results["signals"]:
            sig_lines.append(f"🕹️ MEXC Signals @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            sig_lines.append("**Buy/Inverse Signals**")
            for s in results["signals"]:
                sig_lines.append(fmt_signal_line(s))
        if results["exit_signals"]:
            if not sig_lines:
                sig_lines.append(f"🕹️ MEXC Signals @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            sig_lines.append("**Exit Signals**")
            for x in results["exit_signals"]:
                sig_lines.append(fmt_exit_line(x))
        if sig_lines:
            _discord_post(sig_hook, "\n".join(sig_lines))

        # Trades channel: ONLY actual executed trades
        if trade_events:
            tr_lines = [f"💼 Trades @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"]
            for ev in trade_events:
                tr_lines.append(fmt_trade_line(ev))
            _discord_post(trd_hook, "\n".join(tr_lines))

# ================== Entrypoint ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)

    # Env ${VAR} expansion
    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o
    cfg = expand_env(cfg)

    run(cfg)
