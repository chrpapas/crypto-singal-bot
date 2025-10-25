#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEXC Spot Scanner/Trader

What's inside
-------------
• Top-100-by-marketcap scan (CoinMarketCap) -> normal day/swing/trend signals
• Legacy Movers scan (≥15% 24h move, ≥$5M vol, EMA20>50>100, RSI>=55, Vol>=SMA20) — fakeout-resistant
• Stablecoins excluded
• Paper & live trading (paper default), position limit, risk sizing
• Performance: open/closed trades, OHLC-based exit eval (stop/t1/t2), snapshot print
• Discord:
    - DISCORD_SIGNALS_WEBHOOK : signals only (Top100 section + Movers section). No “skipped/executed” phrasing.
    - DISCORD_TRADES_WEBHOOK  : only posts if trades were executed this run.

Config (YAML)
-------------
persistence.redis_url / key_prefix / ttl_minutes
trading.paper / base_balance_usdt / risk_per_trade_pct / max_concurrent_positions / min_usdt_order / live_market_slippage_bps
movers.limit / min_change_24h / min_volume_usd_24h / cmc_api_key (or env CMC_API_KEY)
performance.enabled / tp_priority / assume_fills / max_bars_eval.{day,swing,trend}
quality.min_avg_dollar_vol_1h

Environment
-----------
CMC_API_KEY
DISCORD_SIGNALS_WEBHOOK
DISCORD_TRADES_WEBHOOK
MEXC_API_KEY, MEXC_SECRET (optional for live)
REDIS_URL (optional if set in config)
"""

import argparse, json, math, os, sys, yaml, requests, redis
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import ccxt

# ----------------------- Utils -----------------------
def to_iso(ts) -> str:
    if ts is None:
        return pd.Timestamp.utcnow().isoformat()
    if isinstance(ts, str):
        return ts
    try:
        return pd.to_datetime(ts, utc=True).isoformat()
    except Exception:
        return pd.Timestamp.utcnow().isoformat()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def body(df): return (df["close"] - df["open"]).abs()

# -------------------- Stablecoins --------------------
STABLE_BASES = {
    "USDT","USDC","FDUSD","TUSD","DAI","USDD","USDP","GUSD","LUSD","SUSD","USDJ","USD1","USTC"
}
# treat gold tokens as “stable-like” for scanning
GOLD_LIKE = {"PAXG", "XAUT", "XAUt"}

def is_stable_like(symbol: str) -> bool:
    try:
        base = symbol.split("/")[0].upper()
    except Exception:
        return False
    return base in STABLE_BASES or base in GOLD_LIKE

# -------------------- Exchange -----------------------
class ExClient:
    def __init__(self):
        self.name = "mexc"
        self.ex = ccxt.mexc({"enableRateLimit": True})
        key, sec = os.environ.get("MEXC_API_KEY"), os.environ.get("MEXC_SECRET")
        if key and sec:
            self.ex.apiKey = key; self.ex.secret = sec
        self._markets = None
    def load_markets(self):
        if self._markets is None:
            try: self._markets = self.ex.load_markets()
            except Exception: self._markets = {}
        return self._markets
    def has_pair(self, pair: str) -> bool:
        mkts = self.load_markets() or {}
        return pair in mkts
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

# -------------------- Redis state --------------------
class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url:
            raise RuntimeError("Redis URL missing.")
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        self.prefix = prefix or "spideybot:v1"
        self.ttl = int(ttl_minutes) * 60 if ttl_minutes else 48*3600
        self.r.setex(self.k("selftest"), self.ttl, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl//60}")
    def k(self, *parts): return ":".join([self.prefix, *map(str, parts)])
    # memory (edge dedup)
    def get_mem(self, kind: str, key: str) -> str:
        return self.r.get(self.k("mem", kind, key)) or ""
    def set_mem(self, kind: str, key: str, val: str):
        self.r.setex(self.k("mem", kind, key), self.ttl, val)
    # positions/perf/portfolio
    def load_positions(self) -> Dict[str, Any]:
        t = self.r.get(self.k("state","active_positions"))
        return json.loads(t) if t else {}
    def save_positions(self, d: Dict[str, Any]):
        self.r.set(self.k("state","active_positions"), json.dumps(d))
    def load_perf(self) -> Dict[str, Any]:
        t = self.r.get(self.k("state","performance"))
        return json.loads(t) if t else {"open_trades": [], "closed_trades": []}
    def save_perf(self, d: Dict[str, Any]):
        self.r.set(self.k("state","performance"), json.dumps(d))
    def load_portfolio(self) -> Dict[str, Any]:
        t = self.r.get(self.k("state","portfolio"))
        return json.loads(t) if t else {}
    def save_portfolio(self, d: Dict[str, Any]):
        self.r.set(self.k("state","portfolio"), json.dumps(d))
    def store_closed_csv(self, csv_text: str):
        self.r.set(self.k("perf","closed_csv"), csv_text)

# -------------------- TA params ----------------------
@dataclass
class DayParams:
    lookback_high:int=30
    vol_sma:int=30
    rsi_min:int=52
    rsi_max:int=78
    stop_mode:str="swing"
    atr_mult:float=1.5

@dataclass
class TrendParams:
    ema20:int=20
    ema50:int=50
    ema100:int=100
    pullback_pct_max:float=10.0
    rsi_min:int=50
    rsi_max:int=70
    vol_sma:int=20
    breakout_lookback:int=55
    stop_mode:str="swing"
    atr_mult:float=2.0

# -------------------- Signals ------------------------
def stop_from(df, mode, atr_mult):
    if mode == "atr":
        a = atr(df, 14).iloc[-1]
        a = 0.0 if np.isnan(a) else float(a)
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

def day_signal(df: pd.DataFrame, p: DayParams):
    look, voln = p.lookback_high, p.vol_sma
    if len(df) < max(look, voln) + 5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) and (p.rsi_min <= r.iloc[-1] <= p.rsi_max)
    if not breakout_ok: return None
    entry = float(last["close"]); stop = stop_from(df, p.stop_mode, p.atr_mult)
    return {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
            "level":float(highlvl),"note":"Breakout","event_bar_ts": df.index[-1].isoformat()}

def swing_signal(df: pd.DataFrame, p: Dict[str,Any]):
    need = max(p.get("ema100",100), p.get("vol_sma",20), p.get("breakout_lookback",34)) + 5
    if len(df) < need: return None
    df=df.copy(); df["ema20"]=ema(df["close"],p.get("ema20",20)); df["ema50"]=ema(df["close"],p.get("ema50",50))
    df["ema100"]=ema(df["close"],p.get("ema100",100)); df["volS"]=sma(df["volume"],p.get("vol_sma",20))
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1]>=p.get("rsi_min",50))
    within=abs((last["close"]-last["ema20"])/max(1e-9,last["ema20"])*100)<=p.get("pullback_pct_max",10.0)
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(p.get("breakout_lookback",34)+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]>df["volS"].iloc[-1] if not np.isnan(df["volS"].iloc[-1]) else True)
    if not (aligned and ((within and bounce) or breakout)): return None
    entry=float(last["close"]); stop=stop_from(df,p.get("stop_mode","swing"),p.get("atr_mult",2.0))
    return {"type":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
            "level":float(hl),"note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout"}

def trend_signal(df: pd.DataFrame, p: TrendParams):
    need=max(p.ema100,p.vol_sma,p.breakout_lookback)+5
    if len(df)<need: return None
    df=df.copy(); df["ema20"]=ema(df["close"],p.ema20); df["ema50"]=ema(df["close"],p.ema50)
    df["ema100"]=ema(df["close"],p.ema100); df["volS"]=sma(df["volume"],p.vol_sma)
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1]>=p.rsi_min)
    within=abs((last["close"]-last["ema20"])/max(1e-9,last["ema20"])*100)<=p.pullback_pct_max
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(p.breakout_lookback+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]>df["volS"].iloc[-1] if not np.isnan(df["volS"].iloc[-1]) else True)
    if not (aligned and ((within and bounce) or breakout)): return None
    entry=float(last["close"]); stop=stop_from(df,p.stop_mode,p.atr_mult)
    return {"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
            "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout"}

# -------------------- CMC helpers --------------------
def cmc_listings(api_key: str, limit: int, convert="USD")->List[dict]:
    url="https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers={"X-CMC_PRO_API_KEY":api_key}
    params={"limit":limit,"convert":convert}
    r=requests.get(url,headers=headers,params=params,timeout=25)
    r.raise_for_status()
    return r.json().get("data",[])

def fetch_top100_pairs_on_mexc(client: ExClient, api_key: str) -> List[str]:
    data = cmc_listings(api_key, limit=100)
    out=[]
    for it in data:
        sym=(it.get("symbol") or "").upper()
        if not sym: continue
        pair=f"{sym}/USDT"
        if is_stable_like(pair):  # exclude stable/gold-like
            continue
        if client.has_pair(pair):
            out.append(pair)
    return out

# -------- Legacy Movers (fakeout-resistant) ----------
def fetch_legacy_movers_symbols(cfg: Dict[str,Any], api_key: str) -> List[str]:
    mv = cfg.get("movers", {})
    data = cmc_listings(api_key, limit=mv.get("limit", 500))
    minchg = mv.get("min_change_24h", 15.0)
    minvol = mv.get("min_volume_usd_24h", 5_000_000)
    out=[]
    for it in data:
        sym=(it.get("symbol") or "").upper()
        if not sym: continue
        q=it.get("quote",{}).get("USD",{})
        ch=float(q.get("percent_change_24h") or 0.0)
        vol=float(q.get("volume_24h") or 0.0)
        if ch>=minchg and vol>=minvol:
            out.append(sym)
    return out

def filter_pairs_on_mexc(client: ExClient, syms: List[str]) -> List[str]:
    out=[]
    for s in syms:
        p=f"{s}/USDT"
        if client.has_pair(p) and (not is_stable_like(p)):
            out.append(p)
    return out

# -------------------- Performance --------------------
def pos_key(exchange:str, pair:str, sig_type:str)->str: return f"{exchange}|{pair}|{sig_type}"

def add_open_trade(perf: Dict[str,Any], *, exchange, symbol, tf, sig_type, entry, stop, t1, t2, event_ts):
    risk = max(1e-12, entry - stop)
    perf.setdefault("open_trades", []).append({
        "id": f"{exchange}|{symbol}|{tf}|{sig_type}|{event_ts}",
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": tf,
        "type": sig_type,
        "opened_at": to_iso(event_ts),
        "entry": float(entry),
        "stop": float(stop),
        "t1": float(t1) if t1 is not None else None,
        "t2": float(t2) if t2 is not None else None,
        "risk": float(risk),
        "status": "open",
    })

def close_trade(tr: Dict[str,Any], *, outcome, price, closed_at, reason):
    tr["status"]="closed"
    tr["closed_at"]=to_iso(closed_at)
    tr["exit_price"]=float(price)
    tr["outcome"]=outcome
    risk=max(1e-12, tr.get("risk", 1e-12))
    rr=(price - tr["entry"]) / risk
    tr["r_multiple"]=float(rr)
    tr["pct_return"]=float((price/tr["entry"] - 1) * 100.0)
    tr["reason"]=reason

def fetch_since(ex_client: ExClient, pair: str, tf: str, since_ts: pd.Timestamp) -> List[dict]:
    lim = 1000 if tf in ("1m","5m","15m","30m") else 420
    df = ex_client.ohlcv(pair, tf, lim)
    df = df.loc[pd.to_datetime(since_ts, utc=True):]
    out=[]
    for ts,row in df.iterrows():
        out.append({"ts": ts, "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])})
    return out

def _eval_hit_order(rows, entry, stop, t1, t2, *, tp_priority="target_first"):
    for r in rows:
        hi, lo = r["high"], r["low"]
        hit_stop = lo <= stop
        hit_t2 = (t2 is not None) and (hi >= t2)
        hit_t1 = (t1 is not None) and (hi >= t1)
        if hit_stop and (hit_t2 or hit_t1):
            if tp_priority == "target_first":
                if hit_t2: return "t2", t2, r["ts"]
                if hit_t1: return "t1", t1, r["ts"]
                return "stop", stop, r["ts"]
            else:
                return "stop", stop, r["ts"]
        if hit_stop: return "stop", stop, r["ts"]
        if hit_t2:   return "t2", t2, r["ts"]
        if hit_t1:   return "t1", t1, r["ts"]
    return None, None, None

def evaluate_open_trades(perf: Dict[str,Any], ex_client: ExClient, cfg: Dict[str,Any]):
    ps = perf.setdefault("open_trades", [])
    if not ps: return
    tp_priority = (cfg.get("performance", {}).get("tp_priority") or "target_first")
    assume = (cfg.get("performance", {}).get("assume_fills") or "next_close").lower()
    max_eval = cfg.get("performance", {}).get("max_bars_eval", {})
    to_close=[]
    for tr in ps:
        if tr.get("status")!="open": continue
        pair=tr["symbol"]; tf=tr["timeframe"]; entry=tr["entry"]; stop=tr["stop"]; t1=tr.get("t1"); t2=tr.get("t2")
        tf_key = {"1h":"day","4h":"swing","1d":"trend"}.get(tf,"day")
        max_n = int(max_eval.get(tf_key, 180))
        try:
            rows = fetch_since(ex_client, pair, tf, pd.to_datetime(tr["opened_at"], utc=True))
            if len(rows) <= 1:
                continue
            if assume == "signal_close":
                eval_rows = rows[1:]
            else:
                # assume next close entry fill
                if len(rows) >= 2:
                    entry_assumed = rows[1]["close"]
                    tr["entry"] = float(entry_assumed)
                    eval_rows = rows[2:]
                else:
                    eval_rows = rows[1:]
            if max_n and len(eval_rows) > max_n:
                eval_rows = eval_rows[:max_n]
            outcome, px, ts = _eval_hit_order(eval_rows, tr["entry"], tr["stop"], t1, t2, tp_priority=tp_priority)
            if outcome:
                to_close.append((tr, outcome, px, ts, f"hit_{outcome}"))
            elif max_n and len(eval_rows) >= max_n:
                to_close.append((tr, "timeout", eval_rows[-1]["close"], eval_rows[-1]["ts"], "max_bars_timeout"))
        except Exception as e:
            print("[perf] eval err:", e)
    if to_close:
        closed = perf.setdefault("closed_trades", [])
        for tr, outcome, price, ts, reason in to_close:
            close_trade(tr, outcome=outcome, price=price, closed_at=ts, reason=reason)
            closed.append(tr)
        perf["open_trades"] = [t for t in perf["open_trades"] if t.get("status")=="open"]

def write_perf_csv(perf: Dict[str,Any], rds: RedisState):
    try:
        closed = perf.get("closed_trades", [])
        if not closed: return
        df = pd.DataFrame(closed)
        rds.store_closed_csv(df.to_csv(index=False))
    except Exception as e:
        print("[perf] csv err:", e)

# -------------------- Paper account -------------------
def ensure_portfolio(rds: RedisState, trading_cfg: Dict[str,Any]) -> Dict[str,Any]:
    pf = rds.load_portfolio()
    if not pf:
        base = float(trading_cfg.get("base_balance_usdt", 1000.0))
        pf = {"cash_usdt": base, "holdings": {}}
        rds.save_portfolio(pf)
        print(f"[portfolio] init paper balance = {base:.2f} USDT")
    return pf

def compute_qty_for_risk(entry: float, stop: float, equity_usdt: float, risk_pct: float) -> float:
    risk_usdt = equity_usdt * max(0.0, risk_pct) / 100.0
    per_unit = max(1e-9, entry - stop)
    return max(0.0, risk_usdt / per_unit)

def paper_buy(rds: RedisState, pf: Dict[str,Any], symbol: str, price: float, qty: float, min_order: float) -> bool:
    cost = price * qty
    if cost < min_order or cost > pf.get("cash_usdt", 0.0):
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
    if not h or h["qty"]<=0: return 0.0
    proceeds = price * h["qty"]
    pf["cash_usdt"] += proceeds
    pf["holdings"].pop(symbol, None)
    rds.save_portfolio(pf)
    return proceeds

def paper_snapshot(client: ExClient, pf: Dict[str,Any], perf: Dict[str,Any]):
    mv_total = 0.0; pnl_total = 0.0; lines=[]
    for sym, pos in sorted(pf.get("holdings", {}).items()):
        qty=float(pos["qty"]); avg=float(pos["avg"])
        px=client.last_price(sym) or avg
        mv = qty*px; pnl=(px-avg)*qty
        mv_total += mv; pnl_total += pnl
        lines.append((sym, qty, avg, px, pnl, mv))
    equity = pf.get("cash_usdt", 0.0) + mv_total
    exposure = mv_total
    # open R
    open_R = 0.0
    for tr in perf.get("open_trades", []):
        if tr.get("status")!="open": continue
        px = client.last_price(tr["symbol"]) or tr["entry"]
        risk = max(1e-12, tr["risk"])
        open_R += (px - tr["entry"]) / risk
    # closed stats (robust if outcome missing)
    closed = perf.get("closed_trades", [])
    win = avgR = medR = bestR = worstR = pfactor = 0.0
    if closed:
        dfc = pd.DataFrame(closed)
        if "r_multiple" in dfc.columns:
            win = float((dfc["r_multiple"] > 0).mean() * 100.0)
            avgR = float(dfc["r_multiple"].mean())
            medR = float(dfc["r_multiple"].median())
            bestR = float(dfc["r_multiple"].max())
            worstR = float(dfc["r_multiple"].min())
            gains = dfc.loc[dfc["r_multiple"]>0, "r_multiple"].sum()
            losses = -dfc.loc[dfc["r_multiple"]<0, "r_multiple"].sum()
            pfactor = float(gains / losses) if losses > 0 else float("inf")
        elif "outcome" in dfc.columns:
            win = float((dfc["outcome"].isin(["t1","t2"])).mean() * 100.0)
    print("\n--- Paper Performance Snapshot ---")
    print(f"Cash:      {pf.get('cash_usdt',0.0):.2f} USDT")
    print(f"Exposure:  {exposure:.2f} USDT  | Positions: {len(pf.get('holdings',{}))}")
    print(f"Equity:    {equity:.2f} USDT  | Unrealized PnL: {pnl_total:+.2f} USDT")
    print(f"Open R:    {open_R:+.2f} R")
    if closed:
        print(f"Closed n:  {len(closed)} | Win%: {win:.1f}% | AvgR: {avgR:.2f} | MedR: {medR:.2f} | PF: {('inf' if pfactor==float('inf') else f'{pfactor:.2f}')} | Best/Worst R: {bestR:.2f}/{worstR:.2f}")
    if lines:
        lines.sort(key=lambda x: x[4], reverse=True)
        top = lines[:5]
        print("Top PnL positions:")
        for sym, qty, avg, last, pnl, mv in top:
            pct = ((last/avg)-1.0)*100.0 if avg>0 else 0.0
            print(f"  {sym}: qty={qty:.6f} avg={avg:.6f} last={last:.6f} | PnL={pnl:+.2f} USDT ({pct:+.2f}%)")
    print("--- End Snapshot ---\n")

# -------------------- Discord ------------------------
def discord_post(hook: str, content: str):
    if not hook or not content: return
    try:
        requests.post(hook, json={"content":content}, timeout=10)
    except Exception as e:
        print("[discord] err:", e)

def fmt_sig(s: dict) -> str:
    def f(x):
        return "-" if x is None else f"{x:.6f}"
    return (f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s.get('note','')}\n"
            f"  entry `{f(s['entry'])}` stop `{f(s['stop'])}` t1 `{f(s.get('t1'))}` t2 `{f(s.get('t2'))}`")

def dedup_signals(signals: List[dict]) -> List[dict]:
    seen=set(); out=[]
    for s in signals:
        k=(s.get("source",""), s["symbol"], s["timeframe"], round(float(s["entry"]),6), round(float(s["stop"]),6))
        if k in seen: continue
        seen.add(k); out.append(s)
    return out

# -------------------- RUN ----------------------------
def run(cfg: Dict[str,Any]):
    client = ExClient()
    rds = RedisState(
        url=(cfg.get("persistence",{}).get("redis_url") or os.environ.get("REDIS_URL")),
        prefix=cfg.get("persistence",{}).get("key_prefix","spideybot:v1"),
        ttl_minutes=int(cfg.get("persistence",{}).get("ttl_minutes", 2880))
    )

    trading = cfg.get("trading", {})
    paper_mode = bool(trading.get("paper", True))
    min_order = float(trading.get("min_usdt_order", 10.0))
    risk_pct  = float(trading.get("risk_per_trade_pct", 1.0))
    max_pos   = int(trading.get("max_concurrent_positions", 10))
    slip_bps  = int(trading.get("live_market_slippage_bps", 10))

    qcfg  = cfg.get("quality", {})
    pfcfg = cfg.get("performance", {"enabled": True})

    api_key = os.environ.get("CMC_API_KEY") or (cfg.get("movers",{}).get("cmc_api_key"))
    if not api_key:
        raise RuntimeError("CMC_API_KEY missing (env or config).")

    # Discord webhooks from ENV
    signals_hook = os.environ.get("DISCORD_SIGNALS_WEBHOOK","").strip()
    trades_hook  = os.environ.get("DISCORD_TRADES_WEBHOOK","").strip()

    # Load state
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading) if paper_mode else {}

    # ---------- Build scan sets ----------
    top100_pairs = fetch_top100_pairs_on_mexc(client, api_key)
    top100_pairs = [p for p in top100_pairs if client.has_pair(p) and (not is_stable_like(p))]

    mv_syms = fetch_legacy_movers_symbols(cfg, api_key)
    movers_pairs = filter_pairs_on_mexc(client, mv_syms)

    # ---------- Scan ----------
    results = {"signals_top100": [], "signals_movers": []}

    # Params
    dayP   = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = TrendParams(**cfg.get("trend_trade_params", {}))

    # Top100 — day/swing/trend
    for pair in top100_pairs:
        try:
            # 1h DAY
            df1h = client.ohlcv(pair, "1h", 300)
            if qcfg.get("min_avg_dollar_vol_1h"):
                ad = (df1h["close"].iloc[-24:] * df1h["volume"].iloc[-24:]).mean()
                if float(ad) < float(qcfg["min_avg_dollar_vol_1h"]): 
                    pass
                else:
                    sig = day_signal(df1h, dayP)
                    if sig:
                        sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc","source":"top100"})
                        evts = sig["event_bar_ts"]
                        memk = f"entry_edge|top100|{pair}|1h|{sig['type']}"
                        if rds.get_mem("entry_edge", memk) != evts:
                            rds.set_mem("entry_edge", memk, evts)
                            results["signals_top100"].append(sig)
            else:
                sig = day_signal(df1h, dayP)
                if sig:
                    sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc","source":"top100"})
                    evts = sig["event_bar_ts"]
                    memk = f"entry_edge|top100|{pair}|1h|{sig['type']}"
                    if rds.get_mem("entry_edge", memk) != evts:
                        rds.set_mem("entry_edge", memk, evts)
                        results["signals_top100"].append(sig)
        except Exception as e:
            print(f"[scan-day] {pair} err:", e)
        try:
            # 4h SWING
            tf = swingP.get("timeframe","4h")
            df4h = client.ohlcv(pair, tf, 420)
            sig = swing_signal(df4h, swingP)
            if sig:
                sig.update({"symbol":pair,"timeframe":tf,"exchange":"mexc","source":"top100"})
                evts = df4h.index[-1].isoformat()
                memk = f"entry_edge|top100|{pair}|{tf}|{sig['type']}"
                if rds.get_mem("entry_edge", memk) != evts:
                    rds.set_mem("entry_edge", memk, evts)
                    results["signals_top100"].append(sig)
        except Exception as e:
            print(f"[scan-swing] {pair} err:", e)
        try:
            # 1d TREND
            dfd = client.ohlcv(pair, "1d", 320)
            sig = trend_signal(dfd, trnP)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1d","exchange":"mexc","source":"top100"})
                evts = dfd.index[-1].isoformat()
                memk = f"entry_edge|top100|{pair}|1d|{sig['type']}"
                if rds.get_mem("entry_edge", memk) != evts:
                    rds.set_mem("entry_edge", memk, evts)
                    results["signals_top100"].append(sig)
        except Exception as e:
            print(f"[scan-trend] {pair} err:", e)

    # Movers — legacy fakeout-resistant: require trend and momentum confirms
    for pair in movers_pairs:
        try:
            df = client.ohlcv(pair, "1h", 300)
            # Trend & momentum confirms (EMA stack + RSI + Volume vs SMA20)
            e20, e50, e100 = ema(df["close"],20), ema(df["close"],50), ema(df["close"],100)
            r = rsi(df["close"],14); volS=sma(df["volume"],20); last=df.iloc[-1]
            trend_ok = e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1]
            rsi_ok   = r.iloc[-1] >= 55
            vol_ok   = last["volume"] >= (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])
            if not (trend_ok and rsi_ok and vol_ok):
                continue
            sig = day_signal(df, dayP)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc","source":"movers","note":"Mover Trend"})
                evts = sig["event_bar_ts"]
                memk = f"entry_edge|movers|{pair}|1h|{sig['type']}"
                if rds.get_mem("entry_edge", memk) != evts:
                    rds.set_mem("entry_edge", memk, evts)
                    results["signals_movers"].append(sig)
        except Exception as e:
            print(f"[movers] {pair} err:", e)

    # Dedup just in case
    results["signals_top100"] = dedup_signals(results["signals_top100"])
    results["signals_movers"] = dedup_signals(results["signals_movers"])

    # ---------- EXECUTION ----------
    already_open_symbols = set()
    for v in positions.get("active_positions", {}).values():
        already_open_symbols.add(v["symbol"])
    n_open = len(already_open_symbols)

    executed_trades: List[str] = []

    def maybe_execute(sig: dict):
        nonlocal n_open
        sym = sig["symbol"]; tf = sig["timeframe"]; typ = sig["type"]; entry = float(sig["entry"]); stop = float(sig["stop"])
        # do not gate signal posting by any checks (we post later regardless)
        # But for trading, apply guards:
        if sym in already_open_symbols: 
            return False
        if n_open >= max_pos:
            return False
        # size & execute
        if paper_mode:
            equity = pf.get("cash_usdt", 0.0)
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            if qty <= 0 or entry*qty < min_order or entry*qty > equity:
                return False
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if not ok: return False
            executed_trades.append(f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `{entry:.6f}` (paper)")
        else:
            try:
                bal = client.ex.fetch_balance().get("USDT", {}).get("free", 0.0)
                equity = float(bal)
                qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
                px = entry * (1 + slip_bps/10000.0)
                notional = qty * px
                if qty <= 0 or notional < min_order or notional > equity:
                    return False
                client.place_market(sym, "buy", qty)
                executed_trades.append(f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `~{px:.6f}` (live)")
            except Exception as e:
                print(f"[live] order err {sym}:", e)
                return False
        # register position & perf
        positions.setdefault("active_positions", {})[pos_key("mexc", sym, typ)] = {
            "exchange":"mexc","symbol":sym,"type":typ,"entry":entry,"timeframe":tf,"ts":pd.Timestamp.utcnow().isoformat()
        }
        add_open_trade(perf, exchange="mexc", symbol=sym, tf=tf, sig_type=typ, entry=entry, stop=stop,
                       t1=sig.get("t1"), t2=sig.get("t2"), event_ts=sig.get("event_bar_ts"))
        already_open_symbols.add(sym)
        n_open += 1
        return True

    # Execute across both sources (order: Top100 then Movers)
    for s in results["signals_top100"]:
        maybe_execute(s)
    for s in results["signals_movers"]:
        maybe_execute(s)

    # ---------- Evaluate exits / persist ----------
    if pfcfg.get("enabled", True):
        evaluate_open_trades(perf, client, cfg)
        write_perf_csv(perf, rds)

    rds.save_positions(positions)
    rds.save_perf(perf)

    # ---------- Logs ----------
    print(f"=== MEXC Signals @ {pd.Timestamp.utcnow().isoformat()} ===")
    print(f"Scanned — 1h:{len(top100_pairs)}  4h:{len(top100_pairs)}  1d:{len(top100_pairs)}  | movers:{len(movers_pairs)}")
    if results["signals_top100"]:
        print("\n--- Signals (Top100) ---")
        for s in results["signals_top100"]:
            print(f"[SIGNAL] {s['symbol']} {s['timeframe']} — {s['type']} — {s.get('note','')} | entry {s['entry']:.6f} stop {s['stop']:.6f} t1 {s.get('t1')} t2 {s.get('t2')}")
    if results["signals_movers"]:
        print("\n--- Signals (Movers) ---")
        for s in results["signals_movers"]:
            print(f"[MOVER ] {s['symbol']} {s['timeframe']} — {s['type']} — {s.get('note','')} | entry {s['entry']:.6f} stop {s['stop']:.6f} t1 {s.get('t1')} t2 {s.get('t2')}")

    # Paper snapshot each run
    if paper_mode:
        paper_snapshot(client, pf, perf)

    # ---------- Discord ----------
    # Signals: post sections that have content; never indicate skipped/executed; just pure signals.
    sig_blocks=[]
    if results["signals_top100"]:
        lines=["**Signals — Top 100**"]
        for s in results["signals_top100"]:
            lines.append(fmt_sig(s))
        sig_blocks.append("\n".join(lines))
    if results["signals_movers"]:
        lines=["**Signals — Movers**"]
        for s in results["signals_movers"]:
            lines.append(fmt_sig(s))
        sig_blocks.append("\n".join(lines))
    if signals_hook and sig_blocks:
        discord_post(signals_hook, "\n\n".join(sig_blocks))

    # Trades: only if any executed this run
    if trades_hook and executed_trades:
        hdr = f"**Trades @ {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC**"
        discord_post(trades_hook, "\n".join([hdr, *executed_trades]))

# -------------------- Entrypoint ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # env expand ${VAR}
    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o
    cfg = expand_env(cfg)

    run(cfg)
