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

# ================== EDGE helpers ==================
def _cross_down(a_now, a_prev, b_now, b_prev) -> bool:
    return (a_prev >= b_prev) and (a_now < b_now)
def _cross_up(a_now, a_prev, b_now, b_prev) -> bool:
    return (a_prev <= b_prev) and (a_now > b_now)

# ================== DAY entries (edge) ==================
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
    sig = {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
           "level":float(highlvl),"note":"Breakout" if breakout_ok else "Retest-Reclaim",
           "event_bar_ts": df.index[-1].isoformat()}
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
    sig = {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.04,6),"t2":round(entry*1.08,6),
           "level": float(e20.iloc[-1]), "note":"Early-Reversal (EMA20>50 + MACD + RSI)",
           "event_bar_ts": df.index[-1].isoformat()}
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

# ================== State (positions, entry/exit memory, performance) ==================
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

# exit memory
def get_exit_memory(state: Dict[str,Any], key: str) -> str:
    mem = state.setdefault("exit_memory", {})
    return mem.get(key, "")
def set_exit_memory(state: Dict[str,Any], key: str, bar_iso: str):
    mem = state.setdefault("exit_memory", {})
    mem[key] = bar_iso

# entry memory
def get_entry_memory(state: Dict[str,Any], key: str) -> str:
    mem = state.setdefault("entry_memory", {})
    return mem.get(key, "")
def set_entry_memory(state: Dict[str,Any], key: str, bar_iso: str):
    mem = state.setdefault("entry_memory", {})
    mem[key] = bar_iso

# ================== Performance tracking ==================
def perf_state(state: Dict[str,Any]) -> Dict[str,Any]:
    return state.setdefault("performance", {"open_trades": [], "closed_trades": []})
def _now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()

def add_open_trade(state, *, exchange, symbol, tf, sig_type, entry, stop, t1, t2, event_ts):
    ps = perf_state(state)
    risk = max(1e-12, entry - stop)  # avoid zero/neg
    ps["open_trades"].append({
        "id": f"{exchange}|{symbol}|{tf}|{sig_type}|{event_ts}",
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": tf,
        "type": sig_type,           # day/swing/trend
        "opened_at": event_ts,
        "entry": float(entry),
        "stop": float(stop),
        "t1": float(t1) if t1 is not None else None,
        "t2": float(t2) if t2 is not None else None,
        "risk": float(risk),
        "status": "open",
    })
def close_trade(tr, *, outcome, price, closed_at, reason):
    tr["status"] = "closed"
    tr["closed_at"] = str(closed_at)
    tr["exit_price"] = float(price)
    tr["outcome"] = outcome  # "stop", "t1", "t2", "exit_signal", "timeout"
    rr = (price - tr["entry"]) / max(1e-12, tr["risk"])
    tr["r_multiple"] = float(rr)
    tr["pct_return"] = float((price/tr["entry"] - 1) * 100.0)
    tr["reason"] = reason
def write_perf_csv(state, path: str):
    try:
        closed = perf_state(state)["closed_trades"]
        if not closed: return
        df = pd.DataFrame(closed)
        df.to_csv(path, index=False)
    except Exception as e:
        print("[perf] csv write err:", e)

def _eval_hit_order(rows, entry, stop, t1, t2, *, tp_priority="target_first"):
    for r in rows:
        hi, lo = r["high"], r["low"]
        hit_stop = (lo <= stop)
        hit_t2 = (t2 is not None) and (hi >= t2)
        hit_t1 = (t1 is not None) and (hi >= t1)
        if hit_stop and (hit_t2 or hit_t1):
            if tp_priority == "target_first":
                if hit_t2: return "t2", t2
                if hit_t1: return "t1", t1
                return "stop", stop
            else:
                return "stop", stop
        if hit_stop: return "stop", stop
        if hit_t2:   return "t2", t2
        if hit_t1:   return "t1", t1
    return None, None

def fetch_since(ex_client: ExClient, pair: str, tf: str, since_ts: pd.Timestamp) -> List[dict]:
    lim = 1000 if tf in ("1m","5m","15m","30m") else 400
    df = ex_client.ohlcv(pair, tf, lim)
    df = df.loc[since_ts:]
    out=[]
    for ts, row in df.iterrows():
        out.append({"ts": ts, "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])})
    return out

def evaluate_open_trades(state, ex_clients: Dict[str,ExClient], cfg: Dict[str,Any]):
    ps = perf_state(state)
    if not ps["open_trades"]: return
    tp_priority = (cfg.get("performance", {}).get("tp_priority") or "target_first")
    max_eval = cfg.get("performance", {}).get("max_bars_eval", {})
    to_close=[]
    for tr in ps["open_trades"]:
        if tr.get("status") != "open": continue
        ex_name = tr["exchange"]; pair = tr["symbol"]; tf = tr["timeframe"]
        client = ex_clients.get(ex_name)
        if client is None: continue
        tf_key = {"1h":"day", "4h":"swing", "1d":"trend"}.get(tf, "day")
        max_n = int(max_eval.get(tf_key, 180))
        try:
            since = pd.to_datetime(tr["opened_at"], utc=True)
            rows = fetch_since(client, pair, tf, since)
            if len(rows) <= 1:
                continue
            assume = (cfg.get("performance", {}).get("assume_fills") or "next_close").lower()
            if assume == "signal_close":
                eval_rows = rows[1:]
            else:
                if len(rows) >= 2:
                    tr["entry"] = rows[1]["close"]
                    eval_rows = rows[2:]
                else:
                    eval_rows = rows[1:]
            if max_n and len(eval_rows) > max_n:
                eval_rows = eval_rows[:max_n]
            outcome, price = _eval_hit_order(eval_rows, tr["entry"], tr["stop"], tr.get("t1"), tr.get("t2"),
                                             tp_priority=tp_priority)
            if outcome:
                to_close.append((tr, outcome, price, eval_rows[-1]["ts"], f"hit_{outcome}"))
            elif max_n and len(eval_rows) >= max_n:
                to_close.append((tr, "timeout", eval_rows[-1]["close"], eval_rows[-1]["ts"], "max_bars_timeout"))
        except Exception as e:
            print("[perf] eval err:", e)

    for tr, outcome, price, ts, reason in to_close:
        close_trade(tr, outcome=outcome, price=price, closed_at=str(ts), reason=reason)
        perf_state(state)["closed_trades"].append(tr)
    ps["open_trades"] = [t for t in ps["open_trades"] if t.get("status")=="open"]

# ================== Bearish / Quality utils ==================
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

# ======== Merge helpers (FUZZY merge identical setups across exchanges) ========
def _roundf(x, n=6):
    try: return float(round(float(x), n))
    except Exception: return None

def _approx_equal(a: float, b: float, tol_pct: float = 0.005, tol_abs: float = None) -> bool:
    try:
        a = float(a); b = float(b)
    except Exception:
        return False
    if tol_abs is not None and abs(a - b) <= tol_abs:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= tol_pct

def merge_entries_fuzzy(signals: list, tol_pct: float = 0.005, tol_abs: float = None) -> list:
    """
    Group entries across exchanges if they are 'the same setup':
    same (type, symbol, timeframe, note) and prices within tolerance.
    Returns collapsed groups with entry/stop/t1/t2 ranges and exchange list.
    """
    groups = []
    for s in signals or []:
        base = (s.get("type"), s.get("symbol"), s.get("timeframe"), s.get("note",""))
        matched = None
        for g in groups:
            if g["_base"] != base:
                continue
            if _approx_equal(s.get("entry"), g["entry_mid"], tol_pct, tol_abs) and \
               _approx_equal(s.get("stop"),  g["stop_mid"],  tol_pct, tol_abs):
                matched = g
                break
        if matched is None:
            g = {
                "_base": base,
                "type": s.get("type"),
                "symbol": s.get("symbol"),
                "timeframe": s.get("timeframe"),
                "note": s.get("note",""),
                "exchanges": [s.get("exchange")],
                "sd_confluence": bool(s.get("sd_confluence", False)),
                "confidence_vals": [s.get("confidence")] if "confidence" in s else [],
                "entry_vals": [float(s.get("entry"))],
                "stop_vals":  [float(s.get("stop"))],
                "t1_vals":    [float(s.get("t1"))] if s.get("t1") is not None else [],
                "t2_vals":    [float(s.get("t2"))] if s.get("t2") is not None else [],
            }
            g["entry_mid"] = float(s.get("entry"))
            g["stop_mid"]  = float(s.get("stop"))
            groups.append(g)
        else:
            matched["exchanges"].append(s.get("exchange"))
            matched["sd_confluence"] = matched["sd_confluence"] or bool(s.get("sd_confluence", False))
            if "confidence" in s: matched["confidence_vals"].append(s.get("confidence"))
            matched["entry_vals"].append(float(s.get("entry")))
            matched["stop_vals"].append(float(s.get("stop")))
            if s.get("t1") is not None: matched["t1_vals"].append(float(s.get("t1")))
            if s.get("t2") is not None: matched["t2_vals"].append(float(s.get("t2")))
            matched["entry_mid"] = float(np.median(matched["entry_vals"]))
            matched["stop_mid"]  = float(np.median(matched["stop_vals"]))

    out = []
    for g in groups:
        def _rng(vals):
            if not vals: return None, None
            return float(min(vals)), float(max(vals))
        e_min,e_max = _rng(g["entry_vals"]); s_min,s_max = _rng(g["stop_vals"])
        t1_min,t1_max = _rng(g["t1_vals"]);  t2_min,t2_max = _rng(g["t2_vals"])
        conf_clean = [c for c in g["confidence_vals"] if c is not None]
        conf_min = min(conf_clean, default=None)
        conf_max = max(conf_clean, default=None)
        out.append({
            "type": g["type"],
            "symbol": g["symbol"],
            "timeframe": g["timeframe"],
            "note": g["note"],
            "exchanges": sorted(set([x for x in g["exchanges"] if x])),
            "sd_confluence": g["sd_confluence"],
            "entry_range": (e_min, e_max),
            "stop_range":  (s_min, s_max),
            "t1_range":    (t1_min, t1_max) if t1_min is not None else None,
            "t2_range":    (t2_min, t2_max) if t2_min is not None else None,
            "confidence_range": (conf_min, conf_max) if conf_min is not None else None,
        })
    return out

# ================== EXIT HELPERS ==================
def _ema(series, n): return ema(series, n)
def _macd_bear(df):
    line, sig, _ = macd(df["close"])
    return line.iloc[-1] < sig.iloc[-1], (line.iloc[-2] >= sig.iloc[-2]) and (line.iloc[-1] < sig.iloc[-1])

# ---- Day (1h) ----
def day_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 60: return False, "", None
    p_ema = int(cfg.get("ema_break", 20))
    p_from = int(cfg.get("rsi_drop_from", 70))
    p_to   = int(cfg.get("rsi_drop_to",   60))
    macd_need = bool(cfg.get("macd_confirm", True))

    e = _ema(df["close"], p_ema)
    r = rsi(df["close"], 14)
    close_now, close_prev = df["close"].iloc[-1], df["close"].iloc[-2]
    e_now, e_prev = e.iloc[-1], e.iloc[-2]

    cross_down = (close_prev >= e_prev) and (close_now < e_now)
    if not cross_down:
        return False, "", df.index[-1]

    rsi_drop_ok = (r.iloc[-2] >= p_from) and (r.iloc[-1] <= p_to)
    macd_bear, macd_cross = _macd_bear(df)
    macd_ok = macd_bear if not macd_need else macd_cross

    if rsi_drop_ok or macd_ok:
        why = f"EMA{p_ema} cross-down; " + ("RSI drop" if rsi_drop_ok else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def day_exit(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 60: return False, ""
    p_ema = int(cfg.get("ema_break", 20))
    p_to   = int(cfg.get("rsi_drop_to", 60))
    macd_need = bool(cfg.get("macd_confirm", True))

    e = _ema(df["close"], p_ema)
    r = rsi(df["close"], 14)
    macd_bear, macd_cross = _macd_bear(df)

    if (df["close"].iloc[-1] < e.iloc[-1]) and ((r.iloc[-1] <= p_to) or (macd_cross if macd_need else macd_bear)):
        return True, f"Close<EMA{p_ema} & momentum roll-over"
    return False, ""

# ---- Swing (4h) ----
def swing_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 120: return False, "", None
    p_ema = int(cfg.get("ema_break", 50))
    rsi_below = int(cfg.get("rsi_below", 50))
    macd_need = bool(cfg.get("macd_confirm", True))

    e = _ema(df["close"], p_ema)
    r = rsi(df["close"], 14)
    close_now, close_prev = df["close"].iloc[-1], df["close"].iloc[-2]
    e_now, e_prev = e.iloc[-1], e.iloc[-2]
    cross_down = (close_prev >= e_prev) and (close_now < e_now)

    macd_bear, macd_cross = _macd_bear(df)
    macd_ok = macd_cross if macd_need else macd_bear

    if cross_down and (r.iloc[-1] <= rsi_below or macd_ok):
        why = f"EMA{p_ema} cross-down; " + ("RSI<=%d" % rsi_below if r.iloc[-1] <= rsi_below else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def swing_exit(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 120: return False, ""
    p_ema = int(cfg.get("ema_break", 50)); rsi_below = int(cfg.get("rsi_below", 50))
    macd_need = bool(cfg.get("macd_confirm", True))
    e = _ema(df["close"], p_ema); r = rsi(df["close"],14)
    macd_bear, macd_cross = _macd_bear(df)
    if (df["close"].iloc[-1] < e.iloc[-1]) and (r.iloc[-1] <= rsi_below or (macd_cross if macd_need else macd_bear)):
        return True, f"Close<EMA{p_ema} & momentum roll-over"
    return False, ""

# ---- Trend (1d) ----
def trend_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 200: return False, "", None
    need_cross = bool(cfg.get("ema_cross_20_50", True))
    rsi_below = int(cfg.get("rsi_below", 50))
    macd_need = bool(cfg.get("macd_confirm", True))

    e20, e50 = _ema(df["close"],20), _ema(df["close"],50)
    cross20_50 = (e20.iloc[-2] >= e50.iloc[-2]) and (e20.iloc[-1] < e50.iloc[-1]) if need_cross else True
    r = rsi(df["close"], 14)
    macd_bear, macd_cross = _macd_bear(df)
    macd_ok = macd_cross if macd_need else macd_bear

    if cross20_50 and (r.iloc[-1] <= rsi_below or macd_ok):
        why = ("EMA20<EMA50 cross; " if need_cross else "") + ("RSI<=%d" % rsi_below if r.iloc[-1] <= rsi_below else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def trend_exit(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 200: return False, ""
    need_cross = bool(cfg.get("ema_cross_20_50", True))
    rsi_below = int(cfg.get("rsi_below", 50))
    macd_need = bool(cfg.get("macd_confirm", True))
    e20, e50 = _ema(df["close"],20), _ema(df["close"],50)
    cross = (e20.iloc[-1] < e50.iloc[-1]) if not need_cross else ((e20.iloc[-2] >= e50.iloc[-2]) and (e20.iloc[-1] < e50.iloc[-1]))
    r = rsi(df["close"], 14)
    macd_bear, macd_cross = _macd_bear(df)
    if cross and (r.iloc[-1] <= rsi_below or (macd_cross if macd_need else macd_bear)):
        return True, "Trend momentum break"
    return False, ""

# ================== Main run ==================
def run(cfg: Dict[str,Any]):
    ex_names = parse_csv_env(cfg.get("exchanges") or "mexc")
    ex_clients = {name: ExClient(name) for name in ex_names}
    watchlist = parse_csv_env(cfg.get("symbols_watchlist",""))

    dayP = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP = TrendParams(**cfg.get("trend_trade_params", {}))
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    mv_cfg = cfg.get("movers", {"enabled": False})
    exits_cfg = cfg.get("exits", {"enabled": False})
    bearish_cfg = cfg.get("bearish_signals", {"enabled": False})
    qcfg = cfg.get("quality", {})
    pfcfg = cfg.get("performance", {})
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
                except Exception as e:
                    print(f"[zones] {ex_name} {pair} err:", e); zones_cache[(ex_name, pair)]=[]

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

                # Multi-TF confirm
                tf_ok_count = 0; mtf_need = 0
                day_multi = dayP.multi_tf or {}
                if sig and day_multi.get("enabled", False):
                    tfs = day_multi.get("confirm_tfs", ["5m","15m","30m"]); mtf_need = int(day_multi.get("min_confirmations", 3))
                    mdfs = {tf: None for tf in tfs}
                    for tf in tfs:
                        try: mdfs[tf] = client.ohlcv(pair, tf, 200)
                        except Exception as e: print(f"[multi-tf] {ex_name} {pair} {tf} err:", e)
                    for tf in tfs:
                        df_tf = mdfs.get(tf)
                        if df_tf is not None and tf_bull_ok(df_tf,
                                      require_ema_stack=day_multi.get("require_ema_stack", True),
                                      require_macd_bull=day_multi.get("require_macd_bull", True),
                                      min_rsi=day_multi.get("min_rsi", 52)):
                            tf_ok_count += 1
                    if tf_ok_count < mtf_need: sig = None

                # Quality gates
                if sig and allow_day:
                    if not cooldown_ok((ex_name, pair, "1h", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1h"): sig = None
                    if sig and avg_dollar_vol(df1h, 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig = None
                    if sig and qcfg.get("require_two_closes_breakout", True) and "level" in sig and "Breakout" in sig["note"]:
                        if not two_close_confirm(df1h, sig["level"], "above"): sig = None
                    if sig:
                        conf = confidence_score(rsi_ok=True, vol_ok=True, ema_ok=True, sd_ok=bool(sig.get("sd_confluence", False)),
                                                mtf_hits=tf_ok_count, mtf_need=mtf_need)
                        sig["confidence"] = conf
                        if conf < qcfg.get("min_confidence", 70): sig = None

                # Edge-trigger dedup (entries)
                if sig and allow_day and qcfg.get("edge_entries", True):
                    evts = sig.get("event_bar_ts") or df1h.index[-1].isoformat()
                    k = f"{ex_name}|{pair}|1h|{sig['type']}|{sig.get('note','')}"
                    last_ev = get_entry_memory(state, k)
                    if evts == last_ev: sig = None
                    else: set_entry_memory(state, k, evts)

                if sig and allow_day:
                    sig.update({"symbol":pair,"timeframe":"1h","exchange":ex_name})
                    results["signals"].append(sig)
                    _last_signal_ts[(ex_name, pair, "1h", sig['type'])] = pd.Timestamp.utcnow()
                    if pfcfg.get("enabled", True):
                        add_open_trade(state, exchange=ex_name, symbol=pair, tf="1h",
                                       sig_type="day", entry=sig["entry"], stop=sig["stop"],
                                       t1=sig.get("t1"), t2=sig.get("t2"),
                                       event_ts=sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()))

            except Exception as e: print(f"[scan-day] {ex_name} {pair} err:", e)

        # SWING (4h)
        for pair in swing_pairs:
            try:
                tf=swingP.get("timeframe","4h"); df4h=client.ohlcv(pair, tf, 400)
                zones = zones_cache.get((ex_name, pair))
                sig = swing_signal(df4h, swingP, sd_cfg, zones)
                if sig and not cooldown_ok((ex_name, pair, tf, sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), tf): sig=None
                if sig and avg_dollar_vol(client.ohlcv(pair, "1h", 48), 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                if sig and qcfg.get("require_two_closes_breakout", True) and "Breakout" in sig["note"]:
                    if not two_close_confirm(df4h, sig["level"], "above"): sig=None
                if sig:
                    conf = confidence_score(rsi_ok=True, vol_ok=True, ema_ok=True, sd_ok=bool(sig.get("sd_confluence", False)), mtf_hits=0, mtf_need=0)
                    sig["confidence"]=conf
                    if conf < qcfg.get("min_confidence",70): sig=None

                if sig and qcfg.get("edge_entries", True):
                    evts = df4h.index[-1].isoformat()
                    k = f"{ex_name}|{pair}|{tf}|{sig['type']}|{sig.get('note','')}"
                    last_ev = get_entry_memory(state, k)
                    if evts == last_ev: sig = None
                    else: set_entry_memory(state, k, evts)

                if sig:
                    sig.update({"symbol":pair,"timeframe":tf,"exchange":ex_name})
                    results["signals"].append(sig)
                    _last_signal_ts[(ex_name, pair, tf, sig['type'])] = pd.Timestamp.utcnow()
                    if pfcfg.get("enabled", True):
                        add_open_trade(state, exchange=ex_name, symbol=pair, tf=tf,
                                       sig_type="swing", entry=sig["entry"], stop=sig["stop"],
                                       t1=sig.get("t1"), t2=sig.get("t2"),
                                       event_ts=df4h.index[-1].isoformat())

            except Exception as e: print(f"[scan-swing] {ex_name} {pair} err:", e)

        # TREND (1d)
        for pair in trend_pairs:
            try:
                dfd=client.ohlcv(pair,"1d",300); zones=zones_cache.get((ex_name,pair))
                sig = trend_signal(dfd, trnP, sd_cfg, zones)
                if sig and not cooldown_ok((ex_name, pair, "1d", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1d"): sig=None
                if sig and avg_dollar_vol(client.ohlcv(pair, "1h", 48), 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                if sig and qcfg.get("require_two_closes_breakout", True) and "Breakout" in sig["note"]:
                    if not two_close_confirm(dfd, sig["level"], "above"): sig=None
                if sig:
                    conf = confidence_score(rsi_ok=True, vol_ok=True, ema_ok=True, sd_ok=bool(sig.get("sd_confluence", False)), mtf_hits=0, mtf_need=0)
                    sig["confidence"]=conf
                    if conf < qcfg.get("min_confidence",70): sig=None

                if sig and qcfg.get("edge_entries", True):
                    evts = dfd.index[-1].isoformat()
                    k = f"{ex_name}|{pair}|1d|{sig['type']}|{sig.get('note','')}"
                    last_ev = get_entry_memory(state, k)
                    if evts == last_ev: sig = None
                    else: set_entry_memory(state, k, evts)

                if sig:
                    sig.update({"symbol":pair,"timeframe":"1d","exchange":ex_name})
                    results["signals"].append(sig)
                    _last_signal_ts[(ex_name, pair, "1d", sig['type'])] = pd.Timestamp.utcnow()
                    if pfcfg.get("enabled", True):
                        add_open_trade(state, exchange=ex_name, symbol=pair, tf="1d",
                                       sig_type="trend", entry=sig["entry"], stop=sig["stop"],
                                       t1=sig.get("t1"), t2=sig.get("t2"),
                                       event_ts=dfd.index[-1].isoformat())

            except Exception as e: print(f"[scan-trend] {ex_name} {pair} err:", e)

    # remember entries in position state
    update_state_with_entries(state, results["signals"])

    # ===== exits (edge-trigger) =====
    if exits_cfg.get("enabled"):
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
                    if exits_cfg.get("edge_trigger", True):
                        ok, why, bar_ts = day_exit_edge(df1h, exits_cfg.get("day", {}))
                    else:
                        ok, why = day_exit(df1h, exits_cfg.get("day", {})); bar_ts = df1h.index[-1]
                    if ok and exits_cfg.get("day", {}).get("multi_tf", {}).get("enabled", True):
                        mtf = exits_cfg["day"]["multi_tf"]; tfs = mtf.get("confirm_tfs", ["5m","15m","30m"])
                        need = int(mtf.get("min_confirmations", 2)); bears = 0
                        for tf in tfs:
                            try:
                                df_tf = client.ohlcv(pair, tf, 200)
                                if tf_bear_ok(df_tf, require_ema_bear=mtf.get("require_ema_bear", True),
                                              require_macd_bear=mtf.get("require_macd_bear", True),
                                              max_rsi=mtf.get("max_rsi", 50)):
                                    bears += 1
                            except Exception: ...
                        if bears < need:
                            ok=False; why = why+" (no multi-TF bearish confirm)"
                    if ok:
                        k = f"{ex_name}|{pair}|1h|day_exit"
                        last_bar = get_exit_memory(state, k)
                        if str(bar_ts) != last_bar:
                            results["exit_signals"].append({"type":"day_exit","reason":why,"symbol":pair,"timeframe":"1h","exchange":ex_name})
                            remove_position(state, ex_name, pair, "day")
                            set_exit_memory(state, k, str(bar_ts))
                            if pfcfg.get("enabled", True) and pfcfg.get("use_exit_signals", True):
                                ps = perf_state(state)
                                for tr in list(ps["open_trades"]):
                                    if tr["exchange"]==ex_name and tr["symbol"]==pair and tr["timeframe"]=="1h":
                                        px = df1h["close"].iloc[-1]
                                        close_trade(tr, outcome="exit_signal", price=px, closed_at=df1h.index[-1], reason="day_exit")
                                        perf_state(state)["closed_trades"].append(tr)
                                ps["open_trades"] = [t for t in ps["open_trades"] if t.get("status")=="open"]

                    # Swing exit (4h)
                    df4h = client.ohlcv(pair,"4h",300)
                    if exits_cfg.get("edge_trigger", True):
                        ok, why, bar_ts = swing_exit_edge(df4h, exits_cfg.get("swing", {}))
                    else:
                        ok, why = swing_exit(df4h, exits_cfg.get("swing", {})); bar_ts = df4h.index[-1]
                    if ok:
                        k = f"{ex_name}|{pair}|4h|swing_exit"; last_bar = get_exit_memory(state, k)
                        if str(bar_ts) != last_bar:
                            results["exit_signals"].append({"type":"swing_exit","reason":why,"symbol":pair,"timeframe":"4h","exchange":ex_name})
                            remove_position(state, ex_name, pair, "swing")
                            set_exit_memory(state, k, str(bar_ts))
                            if pfcfg.get("enabled", True) and pfcfg.get("use_exit_signals", True):
                                ps = perf_state(state)
                                for tr in list(ps["open_trades"]):
                                    if tr["exchange"]==ex_name and tr["symbol"]==pair and tr["timeframe"]=="4h":
                                        px = df4h["close"].iloc[-1]
                                        close_trade(tr, outcome="exit_signal", price=px, closed_at=df4h.index[-1], reason="swing_exit")
                                        perf_state(state)["closed_trades"].append(tr)
                                ps["open_trades"] = [t for t in ps["open_trades"] if t.get("status")=="open"]

                    # Trend exit (1d)
                    dfd = client.ohlcv(pair,"1d",320)
                    if exits_cfg.get("edge_trigger", True):
                        ok, why, bar_ts = trend_exit_edge(dfd, exits_cfg.get("trend", {}))
                    else:
                        ok, why = trend_exit(dfd, exits_cfg.get("trend", {})); bar_ts = dfd.index[-1]
                    if ok:
                        k = f"{ex_name}|{pair}|1d|trend_exit"; last_bar = get_exit_memory(state, k)
                        if str(bar_ts) != last_bar:
                            results["exit_signals"].append({"type":"trend_exit","reason":why,"symbol":pair,"timeframe":"1d","exchange":ex_name})
                            remove_position(state, ex_name, pair, "trend")
                            set_exit_memory(state, k, str(bar_ts))
                            if pfcfg.get("enabled", True) and pfcfg.get("use_exit_signals", True):
                                ps = perf_state(state)
                                for tr in list(ps["open_trades"]):
                                    if tr["exchange"]==ex_name and tr["symbol"]==pair and tr["timeframe"]=="1d":
                                        px = dfd["close"].iloc[-1]
                                        close_trade(tr, outcome="exit_signal", price=px, closed_at=dfd.index[-1], reason="trend_exit")
                                        perf_state(state)["closed_trades"].append(tr)
                                ps["open_trades"] = [t for t in ps["open_trades"] if t.get("status")=="open"]

                except Exception as e: print(f"[exits] {ex_name} {pair} err:", e)

    # ===== bearish setups (sell alerts) =====
    if bearish_cfg.get("enabled", False):
        for ex_name, client in ex_clients.items():
            base_pairs = [p for p in watchlist if client.has_pair(p)]
            day_pairs   = list(dict.fromkeys(base_pairs + movers_pairs_by_ex.get(ex_name, [])))
            swing_pairs = list(dict.fromkeys(base_pairs + movers_pairs_by_ex.get(ex_name, [])))
            trend_pairs = base_pairs

            for pair in day_pairs:
                try:
                    df1h = client.ohlcv(pair, bearish_cfg.get("day", {}).get("timeframe","1h"), 300)
                    zones = zones_cache.get((ex_name, pair))
                    sig = day_bearish_signal(df1h, bearish_cfg, sd_cfg, zones)
                    if sig and qcfg.get("require_two_closes_breakout", True):
                        if not two_close_confirm(df1h, sig["level"], "below"): sig=None
                    if sig and avg_dollar_vol(df1h, 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                    if sig and not cooldown_ok((ex_name, pair, "1h", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1h"): sig=None
                    if sig:
                        sig.update({"symbol":pair,"timeframe":"1h","exchange":ex_name})
                        results["sell_signals"].append(sig)
                        _last_signal_ts[(ex_name, pair, "1h", sig['type'])] = pd.Timestamp.utcnow()
                except Exception as e: print(f"[bear-day] {ex_name} {pair} err:", e)

            for pair in swing_pairs:
                try:
                    tf = bearish_cfg.get("swing", {}).get("timeframe","4h")
                    df4h = client.ohlcv(pair, tf, 400)
                    sig = swing_bearish_signal(df4h, bearish_cfg)
                    if sig and avg_dollar_vol(client.ohlcv(pair, "1h", 48), 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                    if sig and not cooldown_ok((ex_name, pair, tf, sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), tf): sig=None
                    if sig:
                        sig.update({"symbol":pair,"timeframe":tf,"exchange":ex_name})
                        results["sell_signals"].append(sig)
                        _last_signal_ts[(ex_name, pair, tf, sig['type'])] = pd.Timestamp.utcnow()
                except Exception as e: print(f"[bear-swing] {ex_name} {pair} err:", e)

            for pair in trend_pairs:
                try:
                    dfd = client.ohlcv(pair, bearish_cfg.get("trend", {}).get("timeframe","1d"), 320)
                    sig = trend_bearish_signal(dfd, bearish_cfg)
                    if sig and avg_dollar_vol(client.ohlcv(pair, "1h", 48), 24) < qcfg.get("min_avg_dollar_vol_1h", 200000): sig=None
                    if sig and not cooldown_ok((ex_name, pair, "1d", sig['type']), qcfg.get("signal_cooldown_bars", 6), pd.Timestamp.utcnow(), "1d"): sig=None
                    if sig:
                        sig.update({"symbol":pair,"timeframe":"1d","exchange":ex_name})
                        results["sell_signals"].append(sig)
                        _last_signal_ts[(ex_name, pair, "1d", sig['type'])] = pd.Timestamp.utcnow()
                except Exception as e: print(f"[bear-trend] {ex_name} {pair} err:", e)

    # ===== Performance evaluation & persistence =====
    if pfcfg.get("enabled", True):
        evaluate_open_trades(state, ex_clients, cfg)
        write_perf_csv(state, pfcfg.get("csv_path","perf_trades.csv"))

    # persist state
    save_state(state_path, state)

    # ===== Render logs =====
    tol_bps = (qcfg.get("merge_tolerance_bps", 50))  # basis points (0.50% default)
    entries_view = merge_entries_fuzzy(results.get("signals", []), tol_pct=tol_bps/10000.0)

    # simple merges for exits / sells by exchange+text key
    def _exit_key(s: dict): return (s.get("type"), s.get("symbol"), s.get("timeframe"), s.get("reason",""))
    def _sell_key(s: dict): return (s.get("type"), s.get("symbol"), s.get("timeframe"), s.get("note",""), float(s.get("price")), float(s.get("level")))
    def _merge_by_exchange(signals: list, key_fn):
        merged = {}
        for s in signals or []:
            k = key_fn(s)
            if k not in merged:
                t = s.copy(); t["exchanges"] = [s.get("exchange")]; merged[k] = t
            else:
                merged[k]["exchanges"].append(s.get("exchange"))
        return list(merged.values())

    exits_view   = _merge_by_exchange(results.get("exit_signals", []), _exit_key)
    sells_view   = _merge_by_exchange(results.get("sell_signals", []), _sell_key)

    print(f"=== Crypto Signals @ {results['ts']} ===")
    for ex_name, pairs in results.get("movers", {}).items():
        if pairs:
            print(f"Movers considered on {ex_name}: {', '.join(pairs)}")

    if entries_view:
        for g in entries_view:
            sd_tag = " ✅SD" if g.get("sd_confluence") else ""
            exs = ",".join(g["exchanges"])
            e0,e1 = g["entry_range"]; s0,s1 = g["stop_range"]
            t1rng = g.get("t1_range"); t2rng = g.get("t2_range")
            confrng = g.get("confidence_range")
            def _fmt_rng(a,b):
                if a is None or b is None: return "-"
                if abs(a-b) <= max(0.01, 0.001*max(abs(a),abs(b))):
                    return f"{a:.4f}"
                return f"{a:.4f}–{b:.4f}"
            e_txt = _fmt_rng(e0,e1); s_txt = _fmt_rng(s0,s1)
            t1_txt = _fmt_rng(*t1rng) if t1rng else "-"
            t2_txt = _fmt_rng(*t2rng) if t2rng else "-"
            conf_txt = ""
            if confrng:
                c0,c1 = confrng
                conf_txt = f" (conf {c0:.0f}–{c1:.0f})" if c0!=c1 else f" (conf {c0:.0f})"
            print(
                f"[ENTRY] [{exs}] {g['symbol']} {g['timeframe']} {g['type'].upper()} — {g['note']}{sd_tag}{conf_txt} — "
                f"entry {e_txt} stop {s_txt} t1 {t1_txt} t2 {t2_txt}"
            )
    else:
        print("No entry signals this run.")

    if exits_view:
        for x in exits_view:
            exs = ",".join(sorted(set(x.get("exchanges", []))))
            print(f"[EXIT ] [{exs}] {x['symbol']} {x['timeframe']} — {x['type']} — {x['reason']}")
    else:
        print("No exit signals this run.")

    if sells_view:
        for s in sells_view:
            exs = ",".join(sorted(set(s.get("exchanges", []))))
            print(f"[SELL ] [{exs}] {s['symbol']} {s['timeframe']} — {s['type']} — {s['note']} (price {s['price']}, invalidate>{s['invalidate_above']})")
    else:
        print("No bearish setups this run.")

    if pfcfg.get("enabled", True):
        closed = perf_state(state)["closed_trades"]
        if closed:
            dfp = pd.DataFrame(closed)
            try:
                win = (((dfp["outcome"].isin(["t1","t2"])) | (dfp.get("r_multiple", 0) > 0)).mean()) * 100.0
            except Exception:
                win = ((dfp["outcome"].isin(["t1","t2"])).mean()) * 100.0
            avgR = dfp["r_multiple"].mean() if "r_multiple" in dfp else 0.0
            print(f"[perf] closed trades: {len(dfp)} | win%: {win:.1f}% | avg R: {avgR:.2f}")

    print(json.dumps(results, indent=2))

    # ===== Notifications =====
    if results["signals"] or results["exit_signals"] or results["sell_signals"]:
        # Telegram
        tcfg = cfg.get("telegram", {"enabled": False})
        if tcfg.get("enabled"):
            try:
                parts=[]
                if entries_view:
                    parts.append("*Entries*")
                    for g in entries_view:
                        exs = ", ".join(g["exchanges"])
                        sd_tag=" ✅SD" if g.get("sd_confluence") else ""
                        conf_txt=""
                        if g.get("confidence_range"):
                            c0,c1 = g["confidence_range"]
                            conf_txt = f" (conf {c0:.0f}–{c1:.0f})" if c0!=c1 else f" (conf {c0:.0f})"
                        e0,e1 = g["entry_range"]; s0,s1 = g["stop_range"]
                        parts.append(
                            f"• [{exs}] `{g['symbol']}` *{g['type'].upper()}* {g['timeframe']} — {g['note']}{sd_tag}{conf_txt}\n"
                            f"  entry `{e0:.4f}`–`{e1:.4f}` stop `{s0:.4f}`–`{s1:.4f}`"
                        )
                if exits_view:
                    parts.append("*Exits*")
                    for x in exits_view:
                        exs = ", ".join(sorted(set(x.get("exchanges", []))))
                        parts.append(f"• [{exs}] `{x['symbol']}` {x['timeframe']} — {x['type']} — {x['reason']}")
                if sells_view:
                    parts.append("*Bearish setups*")
                    for s in sells_view:
                        exs = ", ".join(sorted(set(s.get("exchanges", []))))
                        parts.append(f"• [{exs}] `{s['symbol']}` {s['timeframe']} — {s['type']} — {s['note']}\n  price `{s['price']}` invalidate>`{s['invalidate_above']}` level `{s['level']}`")
                requests.post(f"https://api.telegram.org/bot{tcfg['bot_token']}/sendMessage",
                              json={"chat_id":tcfg["chat_id"],"text":"\n".join(parts),"parse_mode":"Markdown"},timeout=10)
            except Exception as e: print("[telegram] err:", e)

        # Discord
        dcfg = cfg.get("discord", {"enabled": False})
        if dcfg.get("enabled"):
            try:
                lines=[]
                if entries_view:
                    lines.append("**Entries**")
                    for g in entries_view:
                        exs = ", ".join(g["exchanges"])
                        sd_tag=" ✅SD" if g.get("sd_confluence") else ""
                        conf_txt=""
                        if g.get("confidence_range"):
                            c0,c1 = g["confidence_range"]
                            conf_txt = f" (conf {c0:.0f}–{c1:.0f})" if c0!=c1 else f" (conf {c0:.0f})"
                        e0,e1 = g["entry_range"]; s0,s1 = g["stop_range"]
                        lines.append(f"**[{exs}] {g['symbol']}** ({g['timeframe']} {g['type'].upper()}) — {g['note']}{sd_tag}{conf_txt}\nentry `{e0:.4f}`–`{e1:.4f}` stop `{s0:.4f}`–`{s1:.4f}`")
                if exits_view:
                    lines.append("**Exits**")
                    for x in exits_view:
                        exs = ", ".join(sorted(set(x.get("exchanges", []))))
                        lines.append(f"**[{exs}] {x['symbol']}** ({x['timeframe']}) — {x['type']} — {x['reason']}")
                if sells_view:
                    lines.append("**Bearish setups**")
                    for s in sells_view:
                        exs = ", ".join(sorted(set(s.get("exchanges", []))))
                        lines.append(f"**[{exs}] {s['symbol']}** ({s['timeframe']} {s['type']}) — {s['note']}\nprice `{s['price']}` invalidate>`{s['invalidate_above']}` level `{s['level']}`")
                if not lines: lines=["No signals this run."]
                requests.post(dcfg["webhook"], json={"content":"\n".join(lines)}, timeout=10)
            except Exception as e: print("[discord] err:", e)

    return results

# ================== Bearish signals (sell) ==================
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
        in_supply = any(z["type"]=="supply" and z["low"]<=float(last["close"])<=z["high"] for z in (zones or []))
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
