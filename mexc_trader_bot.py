#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mexc_trader_bot.py — MEXC spot-only scanner/trader with:
- Top-100-by-marketcap scan (signals)
- Legacy Movers scan (signals)
- Bear-safe inverse flow (bearish detectors -> long inverse 3S/5S tokens on MEXC spot)
- Multi-timeframe confirm for day signals (5m/15m/30m/45m)
- Stablecoin/pegged filtering
- Silent-signal registry (no duplicate setups until closed by t1/stop/timeout)
- Paper/live execution with risk sizing and max-open guard
- Discord: Signals (Top100 + Movers + Bear-Safe to the same hook) and Trades (only if an actual trade executed)
- Performance snapshot for paper and silent-signal dashboard printed to logs (not sent to Discord)

Environment (optional):
  REDIS_URL, DISCORD_SIGNALS_WEBHOOK, DISCORD_TRADES_WEBHOOK, CMC_API_KEY,
  MEXC_API_KEY, MEXC_SECRET

Run:
  python3 mexc_trader_bot.py --config mexc_trader_bot_config.yml
"""

import argparse, json, os, sys, yaml, requests, math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import ccxt
import redis

# =============== TA helpers ===============
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

# =============== Exchange wrapper (MEXC) ===============
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

# =============== Dataclasses (only used keys) ===============
@dataclass
class DayParams:
    lookback_high: int = 30
    vol_sma: int = 30
    rsi_min: int = 52
    rsi_max: int = 78
    atr_mult: float = 1.5
    stop_mode: str = "swing"
    day_mtf: dict = field(default_factory=dict)  # {enabled, confirm_tfs, min_confirmations, require_ema_stack, require_macd_bull, min_rsi}

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

# =============== Redis persistence ===============
class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url: raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL.")
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48*3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")
    def k(self, *parts) -> str: return ":".join([self.prefix, *[str(p) for p in parts]])
    def get(self, *parts) -> Optional[str]: return self.r.get(self.k(*parts))
    def set(self, val: str, *parts): self.r.set(self.k(*parts), val)
    def setex(self, val: str, *parts): self.r.setex(self.k(*parts), self.ttl_seconds, val)
    def load_json(self, *parts, default=None):
        txt = self.get(*parts)
        if not txt: return default if default is not None else {}
        try: return json.loads(txt)
        except Exception: return default if default is not None else {}
    def save_json(self, obj, *parts): self.set(json.dumps(obj), *parts)

# =============== SD zones / helpers ===============
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

# =============== Stablecoin filtering ===============
DEFAULT_STABLES = {
    "USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT","USD1","USDE","USDY",
    "USDP","SUSD","EURS","EURT","PYUSD"
}
# AFTER (good): only filters if the BASE is a stable/pegged token
def is_stable_or_pegged(symbol: str, extra: List[str]) -> bool:
    base, _ = symbol.split("/")
    b = base.upper().replace("3L","").replace("3S","").replace("5L","").replace("5S","")
    extras = {e.upper() for e in (extra or [])}
    return (b in DEFAULT_STABLES) or (b in extras)

# =============== Bullish Signals ===============
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

# =============== Multi-TF confirm helper ===============
def tf_bull_ok(df: pd.DataFrame, *, require_ema_stack=True, require_macd_bull=True, min_rsi=50) -> bool:
    if df is None or len(df) < 60: return False
    e20 = ema(df['close'], 20); e50 = ema(df['close'], 50)
    macd_line, macd_sig, _ = macd(df['close']); r = rsi(df['close'], 14)
    ok=True
    if require_ema_stack: ok &= (e20.iloc[-1] > e50.iloc[-1])
    if require_macd_bull: ok &= (macd_line.iloc[-1] > macd_sig.iloc[-1])
    ok &= (r.iloc[-1] >= min_rsi)
    return bool(ok)

# =============== Legacy Movers universe (CMC) ===============
def fetch_cmc_listings(cfg: Dict[str,Any], limit=500) -> List[dict]:
    headers = {"X-CMC_PRO_API_KEY": os.environ.get("CMC_API_KEY") or cfg.get("movers",{}).get("cmc_api_key","")}
    if not headers["X-CMC_PRO_API_KEY"]: return []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": limit, "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        return r.json().get("data", [])
    except Exception:
        return []

def cmc_top100_symbols(cfg: Dict[str,Any]) -> List[str]:
    data = fetch_cmc_listings(cfg, limit=140)
    data.sort(key=lambda x: x.get("quote",{}).get("USD",{}).get("market_cap", 0), reverse=True)
    return [it["symbol"].upper() for it in data[:110] if "symbol" in it]

def cmc_movers_symbols(cfg: Dict[str,Any]) -> List[str]:
    mv = cfg.get("movers", {})
    data = fetch_cmc_listings(cfg, limit=mv.get("limit", 500))
    out=[]; now=pd.Timestamp.utcnow()
    min_change = mv.get("min_change_24h", 15.0); min_vol = mv.get("min_volume_usd_24h", 5_000_000); max_age = mv.get("max_age_days", 365)
    for it in data:
        sym=it.get("symbol","").upper(); q=it.get("quote",{}).get("USD",{})
        ch=(q.get("percent_change_24h") or 0.0); vol=(q.get("volume_24h") or 0.0)
        date_added=pd.to_datetime(it.get("date_added", now.isoformat()), utc=True); age_days=(now-date_added).days
        if (ch>=min_change) and (vol>=min_vol) and (age_days<=max_age): out.append(sym)
    return out

def filter_pairs_on_mexc(client: ExClient, symbols: List[str], quote: str="USDT", extra_stables: List[str]=None) -> List[str]:
    client.load_markets()
    out=[]
    for sym in symbols:
        pair=f"{sym}/{quote}"
        if client.has_pair(pair) and not is_stable_or_pegged(pair, extra_stables or []):
            out.append(pair)
    return out

# =============== Legacy movers signal (fakeout-resistant) ===============
def legacy_mover_signal(df1h: pd.DataFrame, sd_cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    if df1h is None or len(df1h) < 80: return None
    e20, e50 = ema(df1h["close"],20), ema(df1h["close"],50)
    mac_line, mac_sig, _ = macd(df1h["close"])
    r = rsi(df1h["close"],14); vS = sma(df1h["volume"], 20)
    last = df1h.iloc[-1]
    hl = df1h["high"].iloc[-31:-1].max()
    aligned = (e20.iloc[-1] > e50.iloc[-1]) and (mac_line.iloc[-1] > mac_sig.iloc[-1]) and (55 <= r.iloc[-1] <= 80)
    breakout = (last["close"] > hl) and (last["volume"] >= (vS.iloc[-1] if not np.isnan(vS.iloc[-1]) else last["volume"]))
    if not (aligned and breakout): return None
    entry = float(last["close"]); stop = float(min(df1h["low"].iloc[-10:]))
    return {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
            "level": float(hl), "note": "Mover Trend", "event_bar_ts": df1h.index[-1].isoformat()}

# =============== Bearish detectors + inverse mapping ===============
def day_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    tf_cfg = cfg.get("day", {})
    look = int(tf_cfg.get("lookback_low", 20)); voln = int(tf_cfg.get("vol_sma", 20))
    if len(df) < max(look, voln) + 5: return None
    lowlvl = df["low"].iloc[-(look+1):-1].min(); last = df.iloc[-1]; volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    breakdown = last["close"] < lowlvl if tf_cfg.get("require_breakdown", True) else (last["low"] < lowlvl)
    vol_ok = (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) if tf_cfg.get("require_vol_confirm", True) else True
    rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (breakdown and vol_ok and rsi_weak): return None
    entry = float(last["close"]); stop = float(last["high"])
    return {"type":"bear_day","entry":entry,"stop":stop,"t1":None,"t2":None,
            "level":float(lowlvl),"note":"Bearish breakdown 1h","event_bar_ts": df.index[-1].isoformat()}

def swing_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
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
            "level":float(lowlvl),"note":"Bearish breakdown 4h","event_bar_ts": df.index[-1].isoformat()}

def trend_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
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
            "level":float(lowlvl),"note":"Bearish trend breakdown 1d","event_bar_ts": df.index[-1].isoformat()}

def map_bearish_to_inverse_token(client: ExClient, pair: str, suffixes: List[str]) -> Optional[str]:
    """
    For BASE/USDT, try BASE3S/USDT, BASE5S/USDT, etc. (MEXC spot leveraged tokens)
    """
    base, quote = pair.split("/")
    for suf in suffixes or []:
        alt = f"{base}{suf}/{quote}"
        if client.has_pair(alt): return alt
    return None

# =============== Silent signal registry ===============
def silent_key(symbol: str, tf: str, typ: str, source: str) -> str:
    return f"{symbol}|{tf}|{typ}|{source}"

def record_new_silent(rds: RedisState, sig: Dict[str,Any], source: str):
    book = rds.load_json("state","silent_open", default={})
    k = silent_key(sig["symbol"], sig["timeframe"], sig["type"], source)
    book[k] = {
        "symbol": sig["symbol"], "tf": sig["timeframe"], "type": sig["type"], "source": source,
        "entry": float(sig["entry"]), "stop": float(sig["stop"]),
        "t1": float(sig.get("t1")) if sig.get("t1") is not None else None,
        "t2": float(sig.get("t2")) if sig.get("t2") is not None else None,
        "opened_at": sig.get("event_bar_ts"),
        "status": "open"
    }
    rds.save_json(book, "state","silent_open")

def is_silent_open(rds: RedisState, symbol:str, tf:str, typ:str, source:str) -> bool:
    book = rds.load_json("state","silent_open", default={})
    return silent_key(symbol, tf, typ, source) in book

def close_silent_if_hit(rds: RedisState, client: ExClient, cfg: Dict[str,Any]):
    book = rds.load_json("state","silent_open", default={})
    if not book: return
    changed = False
    for k, tr in list(book.items()):
        if tr.get("status")!="open": continue
        pair = tr["symbol"]; tf = tr["tf"]; stop = tr["stop"]; t1 = tr.get("t1")
        try:
            limit = 400 if tf in ("1h","4h") else 330
            df = client.ohlcv(pair, tf, limit)
            since = pd.to_datetime(tr["opened_at"], utc=True)
            df = df.loc[since:]
            rows = df.iloc[1:]
            outcome=None; price=None
            for _,r in rows.iterrows():
                lo, hi = float(r["low"]), float(r["high"])
                if (lo <= stop) and (t1 is not None and hi >= t1):
                    if hi >= t1: outcome, price = "t1", t1
                    else: outcome, price = "stop", stop
                    break
                if lo <= stop:
                    outcome, price = "stop", stop; break
                if t1 is not None and hi >= t1:
                    outcome, price = "t1", t1; break
            if outcome:
                tr["status"]="closed"; tr["closed_at"]=df.index[-1].isoformat(); tr["outcome"]=outcome; tr["exit_price"]=float(price)
                book[k]=tr; changed=True
        except Exception as e:
            print("[silent] eval err", pair, tf, e)
    if changed:
        rds.save_json(book, "state","silent_open")

def silent_dashboard(rds: RedisState):
    book = rds.load_json("state","silent_open", default={})
    by = {"day": {"open":0,"closed":0,"wins":0}, "swing":{"open":0,"closed":0,"wins":0}, "trend":{"open":0,"closed":0,"wins":0}, "mover":{"open":0,"closed":0,"wins":0}, "bear":{"open":0,"closed":0,"wins":0}}
    for tr in book.values():
        src = tr.get("source","top100")
        bucket = "mover" if src=="movers" else ("bear" if src=="bear" else tr.get("type","day"))
        if tr.get("status")=="open": by[bucket]["open"] += 1
        else:
            by[bucket]["closed"] += 1
            if tr.get("outcome")=="t1": by[bucket]["wins"] += 1
    print("--- Silent Signal Performance (since reset) ---")
    def line(name,k): 
        o=by[k]["open"]; c=by[k]["closed"]; w=by[k]["wins"]; winp=(w/max(1,c))*100.0 if c>0 else 0.0
        print(f"{name:6}: closed {c} | open {o} | win% {winp:.1f}%")
    line("Day","day"); line("Swing","swing"); line("Trend","trend"); line("Mover","mover"); line("Bear","bear")
    print("--- End Silent Performance ---")

# =============== Paper portfolio + trades ===============
def ensure_portfolio(rds: RedisState, trading_cfg: Dict[str,Any]) -> Dict[str,Any]:
    pf = rds.load_json("state","portfolio", default={})
    if not pf:
        base_bal = float(trading_cfg.get("base_balance_usdt", 1000.0))
        pf = {"cash_usdt": base_bal, "holdings": {}}
        rds.save_json(pf, "state","portfolio")
        print(f"[portfolio] init paper balance = {base_bal:.2f} USDT")
    return pf

def compute_qty_for_risk(entry: float, stop: float, equity_usdt: float, risk_pct: float) -> float:
    risk_usdt = equity_usdt * max(0.0, risk_pct) / 100.0
    per_unit_risk = max(1e-9, entry - stop)
    qty = risk_usdt / per_unit_risk
    return max(0.0, qty)

def paper_buy(rds: RedisState, pf: Dict[str,Any], symbol: str, price: float, qty: float, min_order: float) -> bool:
    cost = price * qty
    if cost < min_order or cost > pf["cash_usdt"]: return False
    pf["cash_usdt"] -= cost
    h = pf["holdings"].setdefault(symbol, {"qty":0.0, "avg":0.0})
    new_qty = h["qty"] + qty
    h["avg"] = (h["avg"]*h["qty"] + price*qty) / max(1e-12, new_qty)
    h["qty"] = new_qty
    rds.save_json(pf, "state","portfolio")
    return True

# =============== Discord helpers ===============
def _post_discord(hook: str, text: str):
    if not hook or not text.strip(): return
    try: requests.post(hook, json={"content": text}, timeout=10)
    except Exception as e: print("[discord] post err:", e)

def fmt_signal_lines(title: str, sigs: List[Dict[str,Any]]) -> str:
    if not sigs: return ""
    lines=[f"**{title}**"]
    for s in sigs:
        lines.append(
            f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n"
            f"  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`"
        )
    return "\n".join(lines)

def fmt_trade_lines(trades: List[str]) -> str:
    if not trades: return ""
    return "**Trades**\n" + "\n".join(trades)

# =============== RUN ===============
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

    # Discord hooks
    SIG_HOOK = os.environ.get("DISCORD_SIGNALS_WEBHOOK") or cfg.get("discord", {}).get("signals_webhook", "")
    TRD_HOOK = os.environ.get("DISCORD_TRADES_WEBHOOK")  or cfg.get("discord", {}).get("trades_webhook", "")

    # Filters
    extra_stables = cfg.get("filters", {}).get("extra_stables", [])

    # Params
    dayP   = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = TrendParams(**cfg.get("trend_trade_params", {}))
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    mv_cfg = cfg.get("movers", {"enabled": True})
    bear_cfg = cfg.get("bearish_signals", {"enabled": True})
    bear_mode = cfg.get("bear_mode", {"enabled": True})
    inv_suffix = bear_mode.get("inverse_suffixes", ["3S","5S"])

    # Futures toggle (stub for later)
    fut_cfg = cfg.get("futures", {"enabled": False})
    if fut_cfg.get("enabled"):
        print("[futures] Perp futures enabled in config — execution currently stubbed; spot inverse tokens are used for bear-safe flow.")

    # State
    positions = rds.load_json("state","active_positions", default={})
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}

    # Universe — Top100 & Movers
    top100_syms = cmc_top100_symbols(cfg)
    top100_pairs = filter_pairs_on_mexc(client, top100_syms, "USDT", extra_stables)
    log_universe_stats("Top100", top100_syms, top100_pairs)
    
    movers_syms = cmc_movers_symbols(cfg) if mv_cfg.get("enabled", True) else []
    movers_pairs = filter_pairs_on_mexc(client, movers_syms, mv_cfg.get("quote","USDT"), extra_stables)
    log_universe_stats("Movers(CMC)", movers_syms, movers_pairs)
    
    # Fallbacks if empty/too small
    fb_cfg = cfg.get("fallback", {"enabled": True, "min_pairs": 30, "min_usd_vol": 2_000_000, "max_pairs": 60})
    if fb_cfg.get("enabled", True):
        if len(top100_pairs) < int(fb_cfg.get("min_pairs", 30)):
            top100_pairs = mexc_top_usdt_volume_pairs(
                client,
                max_pairs=int(fb_cfg.get("max_pairs", 60)),
                min_usd_vol=float(fb_cfg.get("min_usd_vol", 2_000_000)),
                extra_stables=extra_stables
            )
            print(f"[universe] Top100 fallback -> using MEXC volume universe ({len(top100_pairs)} pairs)")
        if mv_cfg.get("enabled", True) and len(movers_pairs) == 0:
            movers_pairs = top100_pairs[:]   # reuse volume universe for movers if CMC list empty
            print("[universe] Movers fallback -> reusing MEXC volume universe")

    # Zones cache (optional)
    zones_cache = {}
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        for pair in set(top100_pairs + movers_pairs):
            try:
                zdf = client.ohlcv(pair, ztf, zlook)
                zones_cache[pair] = find_zones(
                    zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                )
            except Exception as e:
                print(f"[zones] {pair} err:", e); zones_cache[pair]=[]

    results_top100: List[Dict[str,Any]] = []
    results_movers: List[Dict[str,Any]] = []
    results_bear:   List[Dict[str,Any]] = []
    trade_lines: List[str] = []

    # ======== Top-100 Scan (bullish) ========
    # DAY (1h)
    for pair in top100_pairs:
        try:
            df = client.ohlcv(pair, "1h", 300)
            zones = zones_cache.get(pair)
            sig = day_signal(df, dayP, sd_cfg, zones)
            if sig and (dayP.day_mtf or {}).get("enabled", True):
                mtf = dayP.day_mtf or {}
                tfs = mtf.get("confirm_tfs", ["5m","15m","30m","45m"])
                need = int(mtf.get("min_confirmations", 2))
                ok = 0
                for tf in tfs:
                    try:
                        df_tf = client.ohlcv(pair, tf, 200)
                        if tf_bull_ok(df_tf,
                                      require_ema_stack=mtf.get("require_ema_stack", True),
                                      require_macd_bull=mtf.get("require_macd_bull", True),
                                      min_rsi=mtf.get("min_rsi", 52)):
                            ok += 1
                    except Exception:
                        pass
                if ok < need: sig=None
            if sig:
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc"})
                if not is_silent_open(rds, pair, "1h", sig["type"], "top100"):
                    results_top100.append(sig); record_new_silent(rds, sig, "top100")
        except Exception as e: print(f"[scan-top100-day] {pair} err:", e)

    # SWING (4h)
    tf_swing = swingP.get("timeframe","4h")
    for pair in top100_pairs:
        try:
            df4h = client.ohlcv(pair, tf_swing, 400)
            zones = zones_cache.get(pair)
            sig = swing_signal(df4h, swingP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":tf_swing,"exchange":"mexc"})
                if not is_silent_open(rds, pair, tf_swing, sig["type"], "top100"):
                    results_top100.append(sig); record_new_silent(rds, sig, "top100")
        except Exception as e: print(f"[scan-top100-swing] {pair} err:", e)

    # TREND (1d)
    for pair in top100_pairs:
        try:
            dfd = client.ohlcv(pair, "1d", 320)
            zones = zones_cache.get(pair)
            sig = trend_signal(dfd, trnP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1d","exchange":"mexc"})
                if not is_silent_open(rds, pair, "1d", sig["type"], "top100"):
                    results_top100.append(sig); record_new_silent(rds, sig, "top100")
        except Exception as e: print(f"[scan-top100-trend] {pair} err:", e)

    # ======== Movers Scan (legacy logic) ========
    for pair in movers_pairs:
        try:
            df1h = client.ohlcv(pair, "1h", 260)
            sig = legacy_mover_signal(df1h, sd_cfg)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc"})
                if not is_silent_open(rds, pair, "1h", sig["type"], "movers"):
                    results_movers.append(sig); record_new_silent(rds, sig, "movers")
        except Exception as e: print(f"[scan-movers] {pair} err:", e)

    # ======== Bear-safe Scan (bearish -> inverse long) ========
    if bear_cfg.get("enabled", True) and bear_mode.get("enabled", True):
        # DAY bearish -> inverse on 1h
        for pair in top100_pairs:
            try:
                df1h = client.ohlcv(pair, bear_cfg.get("day", {}).get("timeframe","1h"), 300)
                sigb = day_bearish_signal(df1h, bear_cfg)
                if sigb:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        sig = {"type":"inverse","entry":client.last_price(inv) or sigb["entry"],"stop":None,
                               "t1":None,"t2":None,"level":sigb["level"],
                               "note":f"Inverse LONG from bearish {pair}","event_bar_ts": sigb["event_bar_ts"],
                               "symbol":inv,"timeframe":"1h","exchange":"mexc"}
                        if not is_silent_open(rds, inv, "1h", "inverse", "bear"):
                            results_bear.append(sig); record_new_silent(rds, sig, "bear")
            except Exception as e: print(f"[bear-day] {pair} err:", e)
        # SWING bearish -> inverse on 4h
        for pair in top100_pairs:
            try:
                df4h = client.ohlcv(pair, bear_cfg.get("swing", {}).get("timeframe","4h"), 400)
                sigb = swing_bearish_signal(df4h, bear_cfg)
                if sigb:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        sig = {"type":"inverse","entry":client.last_price(inv) or sigb["entry"],"stop":None,
                               "t1":None,"t2":None,"level":sigb["level"],
                               "note":f"Inverse LONG from bearish {pair}","event_bar_ts": sigb["event_bar_ts"],
                               "symbol":inv,"timeframe":"4h","exchange":"mexc"}
                        if not is_silent_open(rds, inv, "4h", "inverse", "bear"):
                            results_bear.append(sig); record_new_silent(rds, sig, "bear")
            except Exception as e: print(f"[bear-swing] {pair} err:", e)
        # TREND bearish -> inverse on 1d
        for pair in top100_pairs:
            try:
                dfd = client.ohlcv(pair, bear_cfg.get("trend", {}).get("timeframe","1d"), 320)
                sigb = trend_bearish_signal(dfd, bear_cfg)
                if sigb:
                    inv = map_bearish_to_inverse_token(client, pair, inv_suffix)
                    if inv:
                        sig = {"type":"inverse","entry":client.last_price(inv) or sigb["entry"],"stop":None,
                               "t1":None,"t2":None,"level":sigb["level"],
                               "note":f"Inverse LONG from bearish {pair}","event_bar_ts": sigb["event_bar_ts"],
                               "symbol":inv,"timeframe":"1d","exchange":"mexc"}
                        if not is_silent_open(rds, inv, "1d", "inverse", "bear"):
                            results_bear.append(sig); record_new_silent(rds, sig, "bear")
            except Exception as e: print(f"[bear-trend] {pair} err:", e)

    # ======== Execution (paper or live) ========
    active = positions.setdefault("active_positions", {})
    already_open_syms = set(v["symbol"] for v in active.values())
    n_open = len(already_open_syms)

    all_sigs = results_top100 + results_movers + results_bear
    for s in all_sigs:
        sym, tf, typ = s["symbol"], s["timeframe"], s["type"]
        entry = float(s["entry"]); stop = float(s["stop"]) if s.get("stop") is not None else max(1e-9, entry*0.85)
        if sym in already_open_syms or n_open >= max_pos: continue
        if paper_mode:
            equity = pf.get("cash_usdt", 0.0)
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if ok:
                trade_lines.append(f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `{entry:.6f}` (paper)")
                active_key = f"mexc|{sym}|{typ}"
                active[active_key] = {"exchange":"mexc","symbol":sym,"type":typ,"entry":entry,"timeframe":tf,"ts":pd.Timestamp.utcnow().isoformat()}
                already_open_syms.add(sym); n_open += 1
        else:
            try:
                bal = client.ex.fetch_balance().get("USDT", {}).get("free", 0.0)
                equity = float(bal); qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
                px = entry * (1 + slip_bps/10000.0)
                notional = qty * px
                if notional >= min_order and notional <= equity:
                    client.place_market(sym, "buy", qty)
                    trade_lines.append(f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `~{px:.6f}` (live)")
                    active_key = f"mexc|{sym}|{typ}"
                    active[active_key] = {"exchange":"mexc","symbol":sym,"type":typ,"entry":entry,"timeframe":tf,"ts":pd.Timestamp.utcnow().isoformat()}
                    already_open_syms.add(sym); n_open += 1
            except Exception as e:
                print("[live] order err", sym, e)

    rds.save_json(positions, "state","active_positions")

    # ======== Evaluate/close silent signals ========
    close_silent_if_hit(rds, client, cfg)

    # ======== Logs ========
    print(f"=== MEXC Signals @ {pd.Timestamp.utcnow().isoformat()} ===")
    print(f"Scanned — 1h:{len(top100_pairs)}  4h:{len(top100_pairs)}  1d:{len(top100_pairs)}  | movers:{len(movers_pairs)}")
    silent_dashboard(rds)

    # ======== Discord ========
    top100_msg = fmt_signal_lines("Signals — Top 100", results_top100)
    movers_msg = fmt_signal_lines("Signals — Movers", results_movers)
    bear_msg   = fmt_signal_lines("Signals — Bear-Safe (Inverse)", results_bear)
    if top100_msg: _post_discord(SIG_HOOK, top100_msg)
    if movers_msg: _post_discord(SIG_HOOK, movers_msg)
    if bear_msg:   _post_discord(SIG_HOOK, bear_msg)
    tr_msg = fmt_trade_lines(trade_lines)
    if tr_msg: _post_discord(TRD_HOOK, tr_msg)

    # ======== Paper snapshot ========
    if paper_mode:
        mv_total = 0.0; pnl_total=0.0
        pf_state = rds.load_json("state","portfolio", default={"cash_usdt":0.0,"holdings":{}})
        for sym, pos in sorted(pf_state.get("holdings", {}).items()):
            qty=float(pos["qty"]); avg=float(pos["avg"]); last=client.last_price(sym) or avg
            mv_total += qty*last; pnl_total += (last-avg)*qty
        equity = pf_state.get("cash_usdt",0.0) + mv_total
        print("--- Paper Performance Snapshot ---")
        print(f"Cash:      {pf_state.get('cash_usdt',0.0):.2f} USDT")
        print(f"Exposure:  {mv_total:.2f} USDT  | Positions: {len(pf_state.get('holdings',{}))}")
        print(f"Equity:    {equity:.2f} USDT  | Unrealized PnL: {pnl_total:+.2f} USDT")
        print("--- End Snapshot ---")


# --- DIAGNOSTICS ---
def log_universe_stats(label, raw_syms, mexc_pairs):
    print(f"[universe] {label}: CMC syms={len(raw_syms)} -> MEXC tradable (ex-stables)={len(mexc_pairs)}")

# --- MEXC FALLBACK UNIVERSE (Top USDT-volume spot pairs) ---
def mexc_top_usdt_volume_pairs(client: ExClient, *, max_pairs=60, min_usd_vol=2_000_000, extra_stables=None):
    extra_stables = extra_stables or []
    try:
        tickers = client.ex.fetch_tickers()
    except Exception as e:
        print("[fallback] fetch_tickers err:", e)
        return []
    items = []
    for sym, t in tickers.items():
        if "/USDT" not in sym:
            continue
        if is_stable_or_pegged(sym, extra_stables):
            continue
        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume")
            last = t.get("last") or t.get("close")
            try:
                qv = (base_v or 0) * (last or 0)
            except Exception:
                qv = 0
        try:
            qv = float(qv or 0)
        except Exception:
            qv = 0.0
        if qv >= float(min_usd_vol):
            items.append((sym, qv))
    items.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s,_ in items[:max_pairs]]
    print(f"[fallback] MEXC top USDT-volume picked {len(pairs)} pairs (min_usd_vol={min_usd_vol})")
    return pairs
# =============== Entrypoint ===============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)

    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"): return os.environ.get(o[2:-1], o)
        return o
    cfg = expand_env(cfg)
    run(cfg)
