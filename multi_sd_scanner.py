#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, yaml, requests
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np
import ccxt

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

# ================== Exchange wrapper ==================
class ExClient:
    def __init__(self, name: str):
        self.name = name
        self.ex = getattr(ccxt, name)({"enableRateLimit": True})
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
    # allow nested config without crashing
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

# ================== SD zones ==================
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

# ================== Stops ==================
def stop_from(df: pd.DataFrame, mode: str, atr_mult: float) -> float:
    if mode == "atr":
        a = atr(df, 14).iloc[-1]; a = 0 if np.isnan(a) else float(a)
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

# ================== DAY signals (breakout + early-reversal) ==================
def day_signal(df: pd.DataFrame, p: DayParams, sd_cfg: Dict[str,Any], zones=None):
    look, voln = p.lookback_high, p.vol_sma
    if len(df) < max(look, voln)+5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14); last = df.iloc[-1]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout = (last["close"]>highlvl) and (last["volume"]>volS.iloc[-1]) and (p.rsi_min<=r.iloc[-1]<=p.rsi_max)
    retrec   = (df["low"].iloc[-1]<=highlvl) and (last["close"]>highlvl) and (last["volume"]>0.8*volS.iloc[-1]) and (r.iloc[-1]>=p.rsi_min)
    if not (breakout or retrec): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry = float(last["close"]); stop = stop_from(df, p.stop_mode, p.atr_mult)
    sig = {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
           "level":float(highlvl),"note":"Breakout" if breakout else "Retest-Reclaim"}
    if sd_cfg.get("enabled"): sig["sd_confluence"] = in_demand(entry, zones)
    return sig

def day_early_reversal_signal(df: pd.DataFrame, p_map: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    er = p_map.get("early_reversal", {}) or {}
    if not er.get("enabled", False): return None
    if len(df) < max(60, p_map.get("vol_sma",20)) + 5: return None
    e20  = ema(df["close"], 20); e50  = ema(df["close"], 50)
    volS = sma(df["volume"], p_map.get("vol_sma", 20)); r = rsi(df["close"], 14)
    macd_line, macd_sig, _ = macd(df["close"])
    last_close = df["close"].iloc[-1]; last_vol = df["volume"].iloc[-1]
    ema_cross = (e20.iloc[-1] > e50.iloc[-1]) and ((e20.iloc[-2] <= e50.iloc[-2]) if er.get("require_ema_cross", True) else True)
    macd_ok = (macd_line.iloc[-1] > macd_sig.iloc[-1]) if er.get("require_macd_cross", True) else True
    rsi_ok = r.iloc[-1] >= er.get("min_rsi", 55)
    ext = abs((last_close - e20.iloc[-1]) / (e20.iloc[-1] + 1e-9) * 100.0); not_extended = ext <= er.get("max_extension_pct", 6)
    vol_ok = last_vol >= er.get("min_vol_mult", 1.0) * (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last_vol)
    ok = ema_cross and macd_ok and rsi_ok and not_extended and vol_ok
    if not ok: return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer") == "require" and not in_demand(float(last_close), zones): return None
    entry = float(last_close); stop  = stop_from(df, p_map.get("stop_mode","swing"), p_map.get("atr_mult",1.5))
    sig = {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.04,6),"t2":round(entry*1.08,6),
           "level": float(e20.iloc[-1]), "note":"Early-Reversal (EMA20>50 + MACD + RSI)"}
    if sd_cfg.get("enabled"): sig["sd_confluence"] = in_demand(entry, zones)
    return sig

# ================== SWING & TREND entries ==================
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
    entry=float(last['close']); stop=stop_from(df,p.get('stop_mode','swing'),p.get('atr_mult',1.8))
    return {"type":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
            "level":float(hl),"note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout",
            "ema20":float(last['ema20']),"ema50":float(last['ema50']),"ema100":float(last['ema100'])}

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
    sig={"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
         "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout",
         "ema20":float(df['ema20'].iloc[-1]),"ema50":float(df['ema50'].iloc[-1]),"ema100":float(df['ema100'].iloc[-1])}
    if sd_cfg.get("enabled"): sig["sd_confluence"]=in_demand(entry, zones)
    return sig

# ================== BTC market filter ==================
def btc_ok(ex: ExClient, dayp: DayParams) -> bool:
    try:
        df = ex.ohlcv(dayp.btc_symbol, "1h", 120)
        return df["close"].iloc[-1] > ema(df["close"], dayp.btc_ema).iloc[-1]
    except Exception:
        return True

# ================== Movers (CMC Top-500 gainers) ==================
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
def filter_pairs_on_exchanges(ex_clients: Dict[str,ExClient], symbols: List[str], quote: str="USDT") -> Dict[str,List[str]]:
    by_ex = {name: [] for name in ex_clients.keys()}
    for name, client in ex_clients.items():
        for sym in symbols:
            pair=f"{sym}/{quote}"
            if client.has_pair(pair): by_ex[name].append(pair)
    return by_ex

# ================== State (positions) ==================
def load_state(path:str)->Dict[str,Any]:
    try:
        if os.path.exists(path):
            with open(path,"r") as f: return json.load(f)
    except Exception as e: print("[state] load err:", e)
    return {"active_positions":{}}
def save_state(path:str, state:Dict[str,Any]):
    try:
        with open(path,"w") as f: json.dump(state,f)
    except Exception as e: print("[state] save err:", e)
def pos_key(exchange:str, pair:str, sig_type:str)->str: return f"{exchange}|{pair}|{sig_type}"
def update_state_with_entries(state:Dict[str,Any], signals:List[Dict[str,Any]]):
    ap = state.setdefault("active_positions",{})
    for s in signals:
        if s["type"] in ("day","swing","trend"):
            k = pos_key(s["exchange"], s["symbol"], s["type"])
            ap[k] = {"exchange":s["exchange"],"symbol":s["symbol"],"type":s["type"],
                     "entry":s["entry"],"timeframe":s["timeframe"],"ts":pd.Timestamp.utcnow().isoformat()}
def remove_position(state:Dict[str,Any], exchange:str, pair:str, sig_type:str):
    ap = state.setdefault("active_positions",{})
    ap.pop(pos_key(exchange,pair,sig_type), None)

# ================== Day Exits (1h) with optional multi-TF ==================
from typing import Tuple
def day_exit(df:pd.DataFrame, cfg:Dict[str,Any])->Tuple[bool,str]:
    if len(df)<50: return False,""
    r = rsi(df['close'],14); e20 = ema(df['close'], cfg.get("ema_break",20))
    line, sig, _ = macd(df['close'])
    conds=[]
    if r.iloc[-2] > cfg.get("rsi_drop_from",70) and r.iloc[-1] < cfg.get("rsi_drop_to",60): conds.append("RSI drop")
    if df['close'].iloc[-1] < e20.iloc[-1]: conds.append("EMA20 break")
    if cfg.get("macd_confirm",True) and (line.iloc[-1] < sig.iloc[-1] and line.iloc[-2] >= sig.iloc[-2]): conds.append("MACD cross")
    return (len(conds)>=2, ", ".join(conds))

def swing_exit(df:pd.DataFrame, cfg:Dict[str,Any])->Tuple[bool,str]:
    if len(df)<80: return False,""
    r=rsi(df['close'],14); e50=ema(df['close'], cfg.get("ema_break",50)); line,sig,_=macd(df['close'])
    conds=[]
    if r.iloc[-1] < cfg.get("rsi_below",50): conds.append("RSI<50")
    if df['close'].iloc[-1] < e50.iloc[-1]: conds.append("EMA50 break")
    if cfg.get("macd_confirm",True) and (line.iloc[-1] < sig.iloc[-1] and line.iloc[-2] >= sig.iloc[-2]): conds.append("MACD cross")
    return (len(conds)>=2, ", ".join(conds))

def trend_exit(df:pd.DataFrame, cfg:Dict[str,Any])->Tuple[bool,str]:
    if len(df)<120: return False,""
    r=rsi(df['close'],14); e20=ema(df['close'],20); e50=ema(df['close'],50); line,sig,_=macd(df['close'])
    conds=[]
    if r.iloc[-1] < cfg.get("rsi_below",50): conds.append("RSI<50")
    if cfg.get("ema_cross_20_50",True) and (e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-2] >= e50.iloc[-2]): conds.append("EMA20<EMA50")
    if cfg.get("macd_confirm",True) and (line.iloc[-1] < sig.iloc[-1] and line.iloc[-2] >= sig.iloc[-2]): conds.append("MACD cross")
    if df['close'].iloc[-1] < e50.iloc[-1]: conds.append("Close<EMA50")
    return (len(conds)>=2, ", ".join(conds))

# ================== Bearish setups (sell alerts) ==================
def tf_bull_ok(df: pd.DataFrame, *, require_ema_stack=True, require_macd_bull=True, min_rsi=50) -> bool:
    if df is None or len(df) < 60: return False
    e20 = ema(df['close'], 20); e50 = ema(df['close'], 50)
    macd_line, macd_sig, _ = macd(df['close']); r = rsi(df['close'], 14)
    ok=True
    if require_ema_stack: ok &= (e20.iloc[-1] > e50.iloc[-1])
    if require_macd_bull: ok &= (macd_line.iloc[-1] > macd_sig.iloc[-1])
    ok &= (r.iloc[-1] >= min_rsi)
    return bool(ok)
def tf_bear_ok(df: pd.DataFrame, *, require_ema_bear=True, require_macd_bear=True, max_rsi=50) -> bool:
    if df is None or len(df) < 60: return False
    e20 = ema(df['close'], 20); e50 = ema(df['close'], 50)
    macd_line, macd_sig, _ = macd(df['close']); r = rsi(df['close'], 14)
    ok=True
    if require_ema_bear: ok &= (e20.iloc[-1] < e50.iloc[-1])
    if require_macd_bear: ok &= (macd_line.iloc[-1] < macd_sig.iloc[-1])
    ok &= (r.iloc[-1] <= max_rsi)
    return bool(ok)
def fetch_multi_tf(ex_client: ExClient, pair: str, tfs: List[str], limit: int = 200) -> Dict[str, pd.DataFrame]:
    out = {}
    for tf in tfs:
        try: out[tf] = ex_client.ohlcv(pair, tf, limit)
        except Exception as e: print(f"[multi-tf] {ex_client.name} {pair} {tf} err:", e); out[tf] = None
    return out

def day_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    tf_cfg = cfg.get("day", {})
    look = int(tf_cfg.get("lookback_low", 20)); voln = int(tf_cfg.get("vol_sma", 20))
    if len(df) < max(look, voln) + 5: return None
    lowlvl = df["low"].iloc[-(look+1):-1].min(); last = df.iloc[-1]; volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    breakdown = last["close"] < lowlvl if tf_cfg.get("require_breakdown", True) else (last["low"] < lowlvl)
    vol_ok = (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) if tf_cfg.get("require_vol_confirm", True) else True
    rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (breakdown and vol_ok and rsi_weak): return None
    if sd_cfg.get("enabled") and tf_cfg.get("sd", {}).get("require_supply", False):
        in_supply = False
        for z in (zones or []):
            if z["type"] == "supply" and z["low"] <= float(last["close"]) <= z["high"]: in_supply = True; break
        if not in_supply: return None
    note = "Breakdown (1h) under lookback low"
    invalidate = float(max(lowlvl, last["close"] + (last["high"] - last["low"]) * 0.5))
    return {"type":"sell_day","note":note,"price":float(last["close"]),"invalidate_above":invalidate,"level":float(lowlvl)}

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
    invalidate = float(e20.iloc[-1])
    return {"type":"sell_swing","note":"Breakdown (4h) with bearish EMA stack","price":float(last["close"]),
            "invalidate_above":invalidate,"level":float(lowlvl)}

def trend_bearish_signal(df: pd.DataFrame, cfg: Dict[str,Any]):
    tf_cfg = cfg.get("trend", {})
    if len(df) < 200: return None
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    r = rsi(df["close"], 14); last = df.iloc[-1]
    cross = (e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-2] >= e50.iloc[-2]) if tf_cfg.get("ema20_below_50", True) else True
    lowlvl = df["low"].iloc[-(tf_cfg.get("lookback_low",55)+1):-1].min()
    breakdown = last["close"] < lowlvl; rsi_weak = r.iloc[-1] <= tf_cfg.get("rsi_max", 50)
    if not (cross and breakdown and rsi_weak): return None
    invalidate = float(e50.iloc[-1])
    return {"type":"sell_trend","note":"Trend shift (1d): EMA20<50 + breakdown","price":float(last["close"]),
            "invalidate_above":invalidate,"level":float(lowlvl)}

# ================== Quality / noise reduction ==================
_last_signal_ts: Dict[Tuple[str,str,str,str], pd.Timestamp] = {}
def avg_dollar_vol(df: pd.DataFrame, n=24) -> float:
    if df is None or len(df) < n: return 0.0
    px = df["close"].iloc[-n:]; vol = df["volume"].iloc[-n:]
    return float((px * vol).mean())
def two_close_confirm(df: pd.DataFrame, level: float, direction: str = "above") -> bool:
    c1, c2 = df["close"].iloc[-1], df["close"].iloc[-2]
    return (c1 > level and c2 > level) if direction=="above" else (c1 < level and c2 < level)
def cooldown_ok(key: tuple, bars_needed: int, now_ts: pd.Timestamp, tf: str) -> bool:
    mins = {"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}.get(tf,60)
    prev = _last_signal_ts.get(key)
    if not prev: return True
    delta_min = (now_ts - prev).total_seconds()/60.0
    return delta_min >= (bars_needed * mins)
def confidence_score(*, rsi_ok, vol_ok, ema_ok, sd_ok, mtf_hits, mtf_need) -> int:
    score = 0
    score += 25 if rsi_ok else 0
    score += 25 if vol_ok else 0
    score += 25 if ema_ok else 0
    score += 15 if sd_ok else 0
    bonus = max(0, (mtf_hits - mtf_need)) * 5
    return min(100, score + bonus)

def parse_csv_env(val):
    if isinstance(val, list): return val
    if isinstance(val, str): return [s.strip() for s in val.split(",") if s.strip()]
    return []

# ================== Main run ==================
def run(cfg: Dict[str,Any]):
    # exchanges & watchlist
    ex_names = parse_csv_env(cfg.get("exchanges") or "mexc")
    ex_clients = {name: ExClient(name) for name in ex_names}
    watchlist = parse_csv_env(cfg.get("symbols_watchlist",""))

    # params
    dayP = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP = TrendParams(**cfg.get("trend_trade_params", {}))
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    mv_cfg = cfg.get("movers", {"enabled": False})
    exits_cfg = cfg.get("exits", {"enabled": False})
    bearish_cfg = cfg.get("bearish_signals", {"enabled": False})
    qcfg = cfg.get("quality", {})
    state_path = exits_cfg.get("state_file","state.json")
    state = load_state(state_path)

    # movers
    movers_pairs_by_ex = {name: [] for name in ex_clients.keys()}
    if mv_cfg.get("enabled"):
        cmc_syms = fetch_cmc_top500_gainers(cfg)
        movers_pairs_by_ex = filter_pairs_on_exchanges(ex_clients, cmc_syms, mv_cfg.get("quote","USDT"))

    # SD zones cache
    zones_cache = {}
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        for ex_name, client in ex_clients.items():
            pairs = set(watchlist + movers_pairs_by_ex.get(ex_name, []))
            for pair in pairs:
                if not client.has_pair(pair): continue
                try:
                    zdf = client.ohlcv(pair, ztf, zlook)
                    zones_cache[(ex_name, pair)] = find_zones(
                        zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                    )
                except Exception as e: print(f"[zones] {ex_name} {pair} err:", e); zones_cache[(ex_name, pair)]=[]

    results = {"ts": pd.Timestamp.utcnow().isoformat(), "signals": [], "exit_signals": [], "sell_signals": [], "movers": movers_pairs_by_ex}

    # ===== entries =====
    for ex_name, client in ex_clients.items():
        base_pairs = [p for p in watchlist if client.has_pair(p)]
        day_pairs   = list(dict.fromkeys(base_pairs + movers_pairs_by_ex.get(ex_name, [])))
        swing_pairs = list(dict.fromkeys(base_pairs + movers_pairs_by_ex.get(ex_name, [])))
        trend_pairs = base_pairs

        # DAY (1h)
        allow_day = (not dayP.btc_filter) or btc_ok(client, dayP)
        for pair in day_pairs:
            try:
                df1h = client.ohlcv(pair, "1h", 300)
                zones = zones_cache.get((ex_name, pair))
                sig = day_signal(df1h, dayP, sd_cfg, zones)
                if not sig:
                    sig = day_early_reversal_signal(df1h, vars(dayP), sd_cfg, zones)

                # Multi-TF confirm for DAY entries
                tf_ok_count = 0; mtf_need = 0
                day_multi = dayP.multi_tf or {}
                if sig and day_multi.get("enabled", False):
                    tfs = day_multi.get("confirm_tfs", ["5m","15m","30m"]); mtf_need = int(day_multi.get("min_confirmations", 3))
                    mdfs = fetch_multi_tf(client, pair, tfs, 200)
                    for tf in tfs:
                        if tf_bull_ok(mdfs.get(tf),
                                      require_ema_stack=day_multi.get("require_ema_stack", True),
                                      require_macd_bull=day_multi.get("require_macd_bull", True),
                                      min_rsi=day_multi.get("min_rsi", 52)):
                            tf_ok_count += 1
                    if tf_ok_count < mtf_need: sig = None

                # Quality gates
                if sig and allow_day:
                    # cooldown
                    if not cooldown_ok((ex_name, pair, "1h", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1h"):
                        sig = None
                    # liquidity (1h dollar vol)
                    if sig:
                        if avg_dollar_vol(df1h, 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig = None
                    # two-close for breakouts
                    if sig and qcfg.get("require_two_closes_breakout", True) and "level" in sig and "Breakout" in sig["note"]:
                        if not two_close_confirm(df1h, sig["level"], "above"): sig = None
                    # confidence
                    if sig:
                        r_ok = True; v_ok = True; e_ok = True; sd_ok = bool(sig.get("sd_confluence", False))
                        conf = confidence_score(rsi_ok=r_ok, vol_ok=v_ok, ema_ok=e_ok, sd_ok=sd_ok, mtf_hits=tf_ok_count, mtf_need=mtf_need)
                        sig["confidence"] = conf
                        if conf < qcfg.get("min_confidence", 70): sig = None

                if sig and allow_day:
                    sig.update({"symbol":pair,"timeframe":"1h","exchange":ex_name})
                    results["signals"].append(sig)
                    _last_signal_ts[(ex_name, pair, "1h", sig['type'])] = pd.Timestamp.utcnow()

            except Exception as e: print(f"[scan-day] {ex_name} {pair} err:", e)

        # SWING (4h)
        for pair in swing_pairs:
            try:
                tf=swingP.get("timeframe","4h"); df4h=client.ohlcv(pair, tf, 400)
                zones = zones_cache.get((ex_name, pair))
                sig = swing_signal(df4h, swingP, sd_cfg, zones)
                # Quality: cooldown, liquidity, two-close (if breakout), confidence
                if sig:
                    if not cooldown_ok((ex_name, pair, tf, sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), tf): sig=None
                if sig:
                    df1h_liq = client.ohlcv(pair, "1h", 48)
                    if avg_dollar_vol(df1h_liq, 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                if sig and qcfg.get("require_two_closes_breakout", True) and "Breakout" in sig["note"]:
                    if not two_close_confirm(df4h, sig["level"], "above"): sig=None
                if sig:
                    conf = confidence_score(rsi_ok=True, vol_ok=True, ema_ok=True, sd_ok=bool(sig.get("sd_confluence", False)), mtf_hits=0, mtf_need=0)
                    sig["confidence"]=conf
                    if conf < qcfg.get("min_confidence",70): sig=None
                if sig:
                    sig.update({"symbol":pair,"timeframe":tf,"exchange":ex_name})
                    results["signals"].append(sig)
                    _last_signal_ts[(ex_name, pair, tf, sig['type'])] = pd.Timestamp.utcnow()
            except Exception as e: print(f"[scan-swing] {ex_name} {pair} err:", e)

        # TREND (1d)
        for pair in trend_pairs:
            try:
                dfd=client.ohlcv(pair,"1d",300); zones=zones_cache.get((ex_name,pair))
                sig = trend_signal(dfd, trnP, sd_cfg, zones)
                if sig:
                    if not cooldown_ok((ex_name, pair, "1d", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1d"): sig=None
                if sig:
                    df1h_liq = client.ohlcv(pair, "1h", 48)
                    if avg_dollar_vol(df1h_liq, 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                if sig and qcfg.get("require_two_closes_breakout", True) and "Breakout" in sig["note"]:
                    if not two_close_confirm(dfd, sig["level"], "above"): sig=None
                if sig:
                    conf = confidence_score(rsi_ok=True, vol_ok=True, ema_ok=True, sd_ok=bool(sig.get("sd_confluence", False)), mtf_hits=0, mtf_need=0)
                    sig["confidence"]=conf
                    if conf < qcfg.get("min_confidence",70): sig=None
                if sig:
                    sig.update({"symbol":pair,"timeframe":"1d","exchange":ex_name})
                    results["signals"].append(sig)
                    _last_signal_ts[(ex_name, pair, "1d", sig['type'])] = pd.Timestamp.utcnow()
            except Exception as e: print(f"[scan-trend] {ex_name} {pair} err:", e)

    # remember entries
    update_state_with_entries(state, results["signals"])

    # ===== exits =====
    if exits_cfg.get("enabled"):
        # active positions + watchlist
        active_pairs: Dict[str,Set[str]] = {}
        for k,v in state.get("active_positions",{}).items():
            active_pairs.setdefault(v["exchange"], set()).add(v["symbol"])
        for ex_name, client in ex_clients.items():
            watch_pairs = [p for p in watchlist if client.has_pair(p)]
            pos_pairs = list(active_pairs.get(ex_name, set()))
            check_pairs = sorted(set(watch_pairs + pos_pairs))
            for pair in check_pairs:
                try:
                    # Day exit (1h)
                    df1h = client.ohlcv(pair,"1h",200)
                    ok, why = day_exit(df1h, exits_cfg.get("day", {}))
                    # multi-TF confirm
                    day_mtf = exits_cfg.get("day", {}).get("multi_tf", {})
                    if ok and day_mtf.get("enabled", True):
                        tfs = day_mtf.get("confirm_tfs", ["5m","15m","30m"])
                        need = int(day_mtf.get("min_confirmations", 3)); bears = 0
                        mdfs = fetch_multi_tf(client, pair, tfs, 200)
                        for tf in tfs:
                            if tf_bear_ok(mdfs.get(tf),
                                          require_ema_bear=day_mtf.get("require_ema_bear", True),
                                          require_macd_bear=day_mtf.get("require_macd_bear", True),
                                          max_rsi=day_mtf.get("max_rsi", 50)):
                                bears += 1
                        if bears < need: ok=False; why = why+" (no multi-TF bearish confirm)"
                    if ok:
                        x={"type":"day_exit","reason":why,"symbol":pair,"timeframe":"1h","exchange":ex_name}
                        results["exit_signals"].append(x); remove_position(state, ex_name, pair, "day")

                    # Swing exit (4h)
                    df4h = client.ohlcv(pair,"4h",300)
                    ok, why = swing_exit(df4h, exits_cfg.get("swing", {}))
                    if ok:
                        x={"type":"swing_exit","reason":why,"symbol":pair,"timeframe":"4h","exchange":ex_name}
                        results["exit_signals"].append(x); remove_position(state, ex_name, pair, "swing")

                    # Trend exit (1d)
                    dfd = client.ohlcv(pair,"1d",320)
                    ok, why = trend_exit(dfd, exits_cfg.get("trend", {}))
                    if ok:
                        x={"type":"trend_exit","reason":why,"symbol":pair,"timeframe":"1d","exchange":ex_name}
                        results["exit_signals"].append(x); remove_position(state, ex_name, pair, "trend")

                except Exception as e: print(f"[exits] {ex_name} {pair} err:", e)

    # ===== bearish setups (sell alerts) =====
    if bearish_cfg.get("enabled", False):
        for ex_name, client in ex_clients.items():
            base_pairs = [p for p in watchlist if client.has_pair(p)]
            day_pairs   = list(dict.fromkeys(base_pairs + movers_pairs_by_ex.get(ex_name, [])))
            swing_pairs = list(dict.fromkeys(base_pairs + movers_pairs_by_ex.get(ex_name, [])))
            trend_pairs = base_pairs

            # DAY bearish
            for pair in day_pairs:
                try:
                    df1h = client.ohlcv(pair, bearish_cfg.get("day", {}).get("timeframe","1h"), 300)
                    zones = zones_cache.get((ex_name, pair))
                    sig = day_bearish_signal(df1h, bearish_cfg, sd_cfg, zones)
                    # two-close breakdown confirm
                    if sig and qcfg.get("require_two_closes_breakout", True):
                        if not two_close_confirm(df1h, sig["level"], "below"): sig=None
                    # liquidity & cooldown
                    if sig:
                        if avg_dollar_vol(df1h, 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                        if not cooldown_ok((ex_name, pair, "1h", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1h"): sig=None
                    # multi-TF bearish confirm
                    mtf = bearish_cfg.get("day", {}).get("multi_tf", {})
                    if sig and mtf.get("enabled", True):
                        tfs = mtf.get("confirm_tfs", ["5m","15m","30m"])
                        need = int(mtf.get("min_confirmations", 3)); bears = 0
                        mdfs = fetch_multi_tf(client, pair, tfs, 200)
                        for tf in tfs:
                            if tf_bear_ok(mdfs.get(tf),
                                          require_ema_bear=mtf.get("require_ema_bear", True),
                                          require_macd_bear=mtf.get("require_macd_bear", True),
                                          max_rsi=mtf.get("max_rsi", 50)):
                                bears += 1
                        if bears < need: sig=None
                    if sig:
                        sig.update({"symbol":pair,"timeframe":"1h","exchange":ex_name})
                        results["sell_signals"].append(sig)
                        _last_signal_ts[(ex_name, pair, "1h", sig['type'])] = pd.Timestamp.utcnow()
                except Exception as e: print(f"[bear-day] {ex_name} {pair} err:", e)

            # SWING bearish
            for pair in swing_pairs:
                try:
                    tf = bearish_cfg.get("swing", {}).get("timeframe","4h")
                    df4h = client.ohlcv(pair, tf, 400)
                    sig = swing_bearish_signal(df4h, bearish_cfg)
                    if sig:
                        if avg_dollar_vol(client.ohlcv(pair, "1h", 48), 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                        if not cooldown_ok((ex_name, pair, tf, sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), tf): sig=None
                        if sig:
                            sig.update({"symbol":pair,"timeframe":tf,"exchange":ex_name})
                            results["sell_signals"].append(sig)
                            _last_signal_ts[(ex_name, pair, tf, sig['type'])] = pd.Timestamp.utcnow()
                except Exception as e: print(f"[bear-swing] {ex_name} {pair} err:", e)

            # TREND bearish
            for pair in trend_pairs:
                try:
                    dfd = client.ohlcv(pair, bearish_cfg.get("trend", {}).get("timeframe","1d"), 320)
                    sig = trend_bearish_signal(dfd, bearish_cfg)
                    if sig:
                        if avg_dollar_vol(client.ohlcv(pair, "1h", 48), 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                        if not cooldown_ok((ex_name, pair, "1d", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1d"): sig=None
                        if sig:
                            sig.update({"symbol":pair,"timeframe":"1d","exchange":ex_name})
                            results["sell_signals"].append(sig)
                            _last_signal_ts[(ex_name, pair, "1d", sig['type'])] = pd.Timestamp.utcnow()
                except Exception as e: print(f"[bear-trend] {ex_name} {pair} err:", e)

    # persist state
    save_state(state_path, state)

    # ===== Logs =====
    print(f"=== Crypto Signals @ {results['ts']} ===")
    for ex_name, pairs in results["movers"].items():
        if pairs: print(f"Movers considered on {ex_name}: {', '.join(pairs)}")

    if results["signals"]:
        for s in results["signals"]:
            sd_tag = " ✅SD" if s.get("sd_confluence") else ""
            conf = f" (conf {s.get('confidence',0)})"
            print(f"[ENTRY] [{s['exchange']}] {s['symbol']} {s['timeframe']} {s['type'].upper()} — {s['note']}{sd_tag}{conf} — "
                  f"entry {s['entry']} stop {s['stop']} t1 {s['t1']} t2 {s['t2']}")
    else:
        print("No entry signals this run.")

    if results["exit_signals"]:
        for x in results["exit_signals"]:
            print(f"[EXIT ] [{x['exchange']}] {x['symbol']} {x['timeframe']} — {x['type']} — {x['reason']}")
    else:
        print("No exit signals this run.")

    if results["sell_signals"]:
        for s in results["sell_signals"]:
            print(f"[SELL ] [{s['exchange']}] {s['symbol']} {s['timeframe']} — {s['type']} — {s['note']} "
                  f"(price {s['price']}, invalidate>{s['invalidate_above']})")
    else:
        print("No bearish setups this run.")

    print(json.dumps(results, indent=2))

    # ===== Notifications =====
    if results["signals"] or results["exit_signals"] or results["sell_signals"]:
        # Telegram
        tcfg = cfg.get("telegram", {"enabled": False})
        if tcfg.get("enabled"):
            try:
                parts=[]
                if results["signals"]:
                    parts.append("*Entries*")
                    for s in results["signals"]:
                        sd_tag=" ✅SD" if s.get("sd_confluence") else ""
                        parts.append(f"• [{s['exchange']}] `{s['symbol']}` *{s['type'].upper()}* {s['timeframe']} — {s['note']}{sd_tag} (conf {s.get('confidence',0)})\n"
                                     f"  entry `{s['entry']}` stop `{s['stop']}` t1 `{s['t1']}` t2 `{s['t2']}`")
                if results["exit_signals"]:
                    parts.append("*Exits*")
                    for x in results["exit_signals"]:
                        parts.append(f"• [{x['exchange']}] `{x['symbol']}` {x['timeframe']} — {x['type']} — {x['reason']}")
                if results["sell_signals"]:
                    parts.append("*Bearish setups*")
                    for s in results["sell_signals"]:
                        parts.append(f"• [{s['exchange']}] `{s['symbol']}` {s['timeframe']} — {s['type']} — {s['note']}\n"
                                     f"  price `{s['price']}` invalidate>`{s['invalidate_above']}` level `{s['level']}`")
                requests.post(f"https://api.telegram.org/bot{tcfg['bot_token']}/sendMessage",
                              json={"chat_id":tcfg["chat_id"],"text":"\n".join(parts),"parse_mode":"Markdown"},timeout=10)
            except Exception as e: print("[telegram] err:", e)

        # Discord
        dcfg = cfg.get("discord", {"enabled": False})
        if dcfg.get("enabled"):
            try:
                lines=[]
                if results["signals"]:
                    lines.append("**Entries**")
                    for s in results["signals"]:
                        sd_tag=" ✅SD" if s.get("sd_confluence") else ""
                        lines.append(f"**[{s['exchange']}] {s['symbol']}** ({s['timeframe']} {s['type'].upper()}) — {s['note']}{sd_tag} (conf {s.get('confidence',0)})\n"
                                     f"entry `{s['entry']}` stop `{s['stop']}` t1 `{s['t1']}` t2 `{s['t2']}`")
                if results["exit_signals"]:
                    lines.append("**Exits**")
                    for x in results["exit_signals"]:
                        lines.append(f"**[{x['exchange']}] {x['symbol']}** ({x['timeframe']}) — {x['type']} — {x['reason']}")
                if results["sell_signals"]:
                    lines.append("**Bearish setups**")
                    for s in results["sell_signals"]:
                        lines.append(f"**[{s['exchange']}] {s['symbol']}** ({s['timeframe']} {s['type']}) — {s['note']}\n"
                                     f"price `{s['price']}` invalidate>`{s['invalidate_above']}` level `{s['level']}`")
                if not lines: lines=["No signals this run."]
                requests.post(dcfg["webhook"], json={"content":"\n".join(lines)}, timeout=10)
            except Exception as e: print("[discord] err:", e)

    return results

# ================== Entrypoint ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)

    # expand ${VAR}
    def expand_env(obj):
        if isinstance(obj, dict): return {k: expand_env(v) for k,v in obj.items()}
        if isinstance(obj, list): return [expand_env(x) for x in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], obj)
        return obj
    cfg = expand_env(cfg)
    run(cfg)
