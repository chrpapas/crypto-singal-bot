#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mexc_bearsafe_bot.py — Spot-only scanner/trader for MEXC with bear-safe mode.
- Bullish entries: normal spot longs.
- Bearish entries: map to inverse “S” tokens (e.g., BTC3S/USDT) and long those.
- Persistence: Redis (positions, performance, entry/exit memories, portfolio).
- Default trading mode: paper (no live orders). Live spot optional.

ENV you likely want:
  WATCHLIST_SYMBOLS="BTC/USDT,ETH/USDT,SOL/USDT"
  CMC_API_KEY=...                 # if movers.enabled: true
  REDIS_URL=...                   # or put in config
  MEXC_API_KEY=..., MEXC_SECRET=... (only if live trading)

Config keys (same as before); symbols_watchlist can be "${WATCHLIST_SYMBOLS}".
"""

import argparse, json, os, sys, yaml, requests, math
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
    early_reversal: dict = field(default_factory=dict)
    multi_tf: dict = field(default_factory=dict)

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
    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])
    def get_mem(self, kind: str, key: str) -> str:
        return self.r.get(self.k("mem", kind, key)) or ""
    def set_mem(self, kind: str, key: str, bar_iso: str):
        self.r.setex(self.k("mem", kind, key), self.ttl_seconds, bar_iso)
    def load_positions(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","active_positions"))
        return json.loads(txt) if txt else {}
    def save_positions(self, d: Dict[str, Any]):
        self.r.set(self.k("state","active_positions"), json.dumps(d))
    def load_perf(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state","performance")); 
        return json.loads(txt) if txt else {"open_trades": [], "closed_trades": []}
    def save_perf(self, d: Dict[str, Any]):
        self.r.set(self.k("state","performance"), json.dumps(d))
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

def day_early_reversal_signal(df: pd.DataFrame, p_map: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    er = (p_map.get("early_reversal") or {})
    if not er.get("enabled", False): return None
    if len(df) < max(60, p_map.get("vol_sma",20)) + 5: return None
    e20  = ema(df["close"], 20); e50  = ema(df["close"], 50)
    volS = sma(df["volume"], p_map.get("vol_sma", 20)); r = rsi(df["close"], 14)
    macd_line, macd_sig, _ = macd(df["close"])
    last_close = df["close"].iloc[-1]; last_vol = df["volume"].iloc[-1]
    prev_cross_ok = (e20.iloc[-2] <= e50.iloc[-2]) if er.get("require_ema_cross", True) else True
    ema_cross = (e20.iloc[-1] > e50.iloc[-1]) and prev_cross_ok
    macd_ok = (macd_line.iloc[-1] > macd_sig.iloc[-1]) if er.get("require_macd_cross", True) else True
    rsi_ok = r.iloc[-1] >= er.get("min_rsi", 55)
    ext = abs((last_close - e20.iloc[-1]) / (e20.iloc[-1] + 1e-9) * 100.0); not_extended = ext <= er.get("max_extension_pct", 6)
    vol_ok = last_vol >= er.get("min_vol_mult", 1.0) * (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last_vol)
    ok = ema_cross and macd_ok and rsi_ok and not_extended and vol_ok
    if not ok: return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer") == "require" and not in_demand(float(last_close), zones): return None
    entry = float(last_close); stop  = stop_from(df, p_map.get("stop_mode","swing"), p_map.get("atr_mult",1.5))
    return {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.04,6),"t2":round(entry*1.08,6),
            "level": float(e20.iloc[-1]), "note":"Early-Reversal (EMA20>50 + MACD + RSI)",
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

# ---- Bearish detectors (generate a bearish state; we’ll convert to inverse) ----
def day_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    tf_cfg = cfg.get("day", {})
    look = int(tf_cfg.get("lookback_low", 20)); voln = int(tf_cfg.get("vol_sma", 20))
    if len(df) < max(look, voln) + 5: return None
    lowlvl = df["low"].iloc[-(look+1):-1].min(); last = df.iloc[-1]; volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    breakdown = last["close"] < lowlvl if tf_cfg.get("require_breakdown", True) else (last["low"] < lowlvl)
    vol_ok = (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) if tf_cfg.get("require_vol_confirm", True) else True
    rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (breakdown and vol_ok and rsi_weak): return None
    return {"type":"bear_day","base_note":"Bearish breakdown 1h"}

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
    return {"type":"bear_swing","base_note":"Bearish breakdown 4h"}

def trend_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]):
    tf_cfg = cfg.get("trend", {})
    if len(df) < 200: return None
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    r = rsi(df["close"], 14); last = df.iloc[-1]
    cross = (e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-2] >= e50.iloc[-2]) if tf_cfg.get("ema20_below_50", True) else True
    lowlvl = df["low"].iloc[-(tf_cfg.get("lookback_low",55)+1):-1].min()
    breakdown = last["close"] < lowlvl; rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (cross and breakdown and rsi_weak): return None
    return {"type":"bear_trend","base_note":"Bearish trend breakdown 1d"}

# ================== BTC market filter ==================
def btc_ok(ex: ExClient, dayp: DayParams) -> bool:
    try:
        df = ex.ohlcv(dayp.btc_symbol, "1h", 120)
        return df["close"].iloc[-1] > ema(df["close"], dayp.btc_ema).iloc[-1]
    except Exception:
        return True

# ================== Movers (CMC Top-500 filtered to MEXC) ==================
def fetch_cmc_top500_gainers(cfg: Dict[str,Any]) -> List[str]:
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
        if (ch>=min_change) and (vol>=min_vol) and (age_days<=max_age): out.append(sym)
    return out

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

def inverse_entry_for(client: ExClient, inv_symbol: str, *, stop_mode="swing", atr_mult=1.5) -> Tuple[float,float]:
    """Build entry/stop from the inverse token's own tape."""
    df = client.ohlcv(inv_symbol, "1h", 200)
    entry = float(df["close"].iloc[-1])
    stop  = stop_from(df, stop_mode, atr_mult)
    return entry, stop

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
        "t1": float(t1) if t1 is not None else None,
        "t2": float(t2) if t2 is not None else None,
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

# ================== Sizing (paper & live) ==================
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

    # Build watchset (from env CSV if provided)
    wl_cfg = cfg.get("symbols_watchlist", [])
    if isinstance(wl_cfg, str):
        wl_str = os.getenv("WATCHLIST_SYMBOLS", wl_cfg)
        watchlist = [s.strip() for s in wl_str.split(",") if s.strip()]
    else:
        # also allow WATCHLIST_SYMBOLS to override list
        env_csv = os.getenv("WATCHLIST_SYMBOLS")
        watchlist = [s.strip() for s in env_csv.split(",")] if env_csv else list(wl_cfg or [])
    watchlist = [p for p in watchlist if client.has_pair(p)]
    if not watchlist:
        print("[watchlist] empty after filtering to MEXC symbols")

    # movers filtered to MEXC
    movers_pairs = []
    if cfg.get("movers", {}).get("enabled", False):
        try:
            cmc_syms = fetch_cmc_top500_gainers(cfg)
            movers_pairs = filter_pairs_on_mexc(client, cmc_syms, cfg.get("movers",{}).get("quote","USDT"))
        except Exception as e:
            print("[movers] err:", e)

    def uniq(seq): return list(dict.fromkeys(seq))
    base_pairs = watchlist
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

    # Params
    dayP   = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = TrendParams(**cfg.get("trend_trade_params", {}))
    bearish_cfg = cfg.get("bearish_signals", {"enabled": False})
    exits_cfg   = cfg.get("exits", {"enabled": True})
    qcfg        = cfg.get("quality", {})
    pfcfg       = cfg.get("performance", {"enabled": True})
    bear_mode   = cfg.get("bear_mode", {"enabled": True})
    inv_suffix  = bear_mode.get("inverse_suffixes", ["3S","5S"])

    # Load states
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}

    results = {"ts": pd.Timestamp.utcnow().isoformat(), "signals": [], "movers": {"mexc": movers_pairs}}

    # ===== entries (bullish) =====
    # DAY (1h)
    allow_day = (not dayP.btc_filter) or btc_ok(client, dayP)
    if not allow_day:
        print("[filter] BTC below EMA — day signals paused")
    for pair in scan_pairs_day:
        try:
            df1h = client.ohlcv(pair, "1h", 300)
            zones = zones_cache.get(pair)
            sig = day_signal(df1h, dayP, sd_cfg, zones)
            if not sig:
                sig = day_early_reversal_signal(df1h, vars(dayP), sd_cfg, zones)
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
                sig_bear = day_bearish_signal(df1h, bearish_cfg, sd_cfg, zones_cache.get(pair))
                if sig_bear:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        entry, stop = inverse_entry_for(client, inv, stop_mode="swing", atr_mult=1.5)
                        sig = {"type":"inverse","entry":entry,"stop":stop,"t1":round(entry*1.04,6),"t2":round(entry*1.08,6),
                               "level":None,"note":f"Inverse LONG from bearish {pair}","event_bar_ts": df1h.index[-1].isoformat(),
                               "symbol":inv,"timeframe":"1h","exchange":"mexc"}
                        results["signals"].append(sig)
                    else:
                        print(f"[bear-safe] No inverse token for {pair}")
            except Exception as e:
                print(f"[bear-day] {pair} err:", e)
        # SWING bearish
        for pair in scan_pairs_swing:
            try:
                df4h = client.ohlcv(pair, bearish_cfg.get("swing", {}).get("timeframe","4h"), 400)
                sig_bear = swing_bearish_signal(df4h, bearish_cfg)
                if sig_bear:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        entry, stop = inverse_entry_for(client, inv, stop_mode="swing", atr_mult=2.0)
                        sig = {"type":"inverse","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
                               "level":None,"note":f"Inverse LONG from bearish {pair}","event_bar_ts": df4h.index[-1].isoformat(),
                               "symbol":inv,"timeframe":"4h","exchange":"mexc"}
                        results["signals"].append(sig)
                    else:
                        print(f"[bear-safe] No inverse token for {pair}")
            except Exception as e:
                print(f"[bear-swing] {pair} err:", e)
        # TREND bearish
        for pair in scan_pairs_trend:
            try:
                dfd = client.ohlcv(pair, bearish_cfg.get("trend", {}).get("timeframe","1d"), 320)
                sig_bear = trend_bearish_signal(dfd, bearish_cfg)
                if sig_bear:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        entry, stop = inverse_entry_for(client, inv, stop_mode="swing", atr_mult=2.0)
                        sig = {"type":"inverse","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
                               "level":None,"note":f"Inverse LONG from bearish {pair}","event_bar_ts": dfd.index[-1].isoformat(),
                               "symbol":inv,"timeframe":"1d","exchange":"mexc"}
                        results["signals"].append(sig)
                    else:
                        print(f"[bear-safe] No inverse token for {pair}")
            except Exception as e:
                print(f"[bear-trend] {pair} err:", e)

    # ===== EXECUTE entries (paper or live) =====
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else rds.load_portfolio()

    active_keys = set(positions.get("active_positions", {}).keys())
    n_open = len(active_keys)

    for sig in results["signals"]:
        sym = sig["symbol"]; tf = sig["timeframe"]; typ = sig["type"]
        entry = float(sig["entry"]); stop = float(sig["stop"])
        # Skip if already holding this symbol
        if any(sym in k for k in positions.get("active_positions",{}).keys()):
            print(f"[trade] skip {sym} — already open")
            continue
        if n_open >= max_pos:
            print(f"[trade] skip {sym} — max positions reached ({max_pos})")
            continue

        if paper_mode:
            equity = float(pf.get("cash_usdt", 0.0))
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if not ok:
                need = entry*qty
                print(f"[paper] skip {sym} — not enough cash or below min order (need≈{need:.2f} USDT)")
                continue
            print(f"[paper] BUY {sym} {qty:.6f} @ {entry:.6f}")
        else:
            try:
                px = entry * (1 + slip_bps/10000.0)
                bal = client.ex.fetch_balance().get("USDT", {}).get("free", 0.0)
                equity = float(bal)
                qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
                notional = qty * px
                if notional < min_order or notional > equity:
                    print(f"[live] skip {sym} — insufficient balance or below min")
                    continue
                client.place_market(sym, "buy", qty)
                print(f"[live] BUY {sym} {qty:.6f} @ ~{px:.6f}")
            except Exception as e:
                print(f"[live] order err {sym}:", e); continue

        # Register position & perf
        signal_for_state = {"exchange":"mexc","symbol":sym,"type":typ,"entry":entry,"timeframe":tf}
        update_positions_active(positions, signal_for_state)
        add_open_trade(perf, exchange="mexc", symbol=sym, tf=tf, sig_type=typ, entry=entry, stop=stop,
                       t1=sig.get("t1"), t2=sig.get("t2"), event_ts=sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()))
        n_open += 1

    # Save states
    rds.save_positions(positions)
    rds.save_perf(perf)

    # ===== Logs =====
    print(f"=== MEXC Signals @ {results['ts']} ===")
    print(f"Scanned — 1h:{len(scan_pairs_day)}  4h:{len(scan_pairs_swing)}  1d:{len(scan_pairs_trend)}  | watchlist:{len(watchlist)}  movers:{len(movers_pairs)}")
    if results["movers"]["mexc"]:
        print("Movers (MEXC):", ", ".join(results["movers"]["mexc"][:20]) + (" ..." if len(results["movers"]["mexc"])>20 else ""))
    if results["signals"]:
        for s in results["signals"]:
            t1 = f"{s['t1']:.6f}" if s.get("t1") else "-"
            t2 = f"{s['t2']:.6f}" if s.get("t2") else "-"
            print(f"[ENTRY] {s['symbol']} {s['timeframe']} — {s['type']} — {s['note']} | entry {s['entry']:.6f} stop {s['stop']:.6f} t1 {t1} t2 {t2}")
    else:
        print("No entry signals.")

# ================== Entrypoint ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)

    # env expansion ${VAR} and WATCHLIST_SYMBOLS handling
    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            key = o[2:-1]
            return os.environ.get(key, o)
        return o
    cfg = expand_env(cfg)

    # Convert symbols_watchlist to list from env CSV if needed
    wl_cfg = cfg.get("symbols_watchlist", [])
    if isinstance(wl_cfg, str):
        wl_str = os.getenv("WATCHLIST_SYMBOLS", wl_cfg)
        cfg["symbols_watchlist"] = [s.strip() for s in wl_str.split(",") if s.strip()]

    run(cfg)
