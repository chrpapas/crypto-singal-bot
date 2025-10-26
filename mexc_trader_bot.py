#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Option A — Full trading bot with:
- Top100 CMC signal scan (no watchlist), Movers legacy scan, stablecoin filter
- Day MTF confirmations (5m/15m/30m/45m)
- Silent-signal positions (no duplicates until t1/stop), with per-setup performance dashboard (console only)
- Paper or live trading with partial exits (Option 2), simple trailing, and position cap
- Separate Discord webhooks for signals vs trades (env: DISCORD_SIGNALS_WEBHOOK, DISCORD_TRADES_WEBHOOK)

Requires: ccxt, pandas, numpy, redis, requests, pyyaml
"""

import argparse, json, os, sys, yaml, requests, math, time
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
    def ticker_last(self, symbol: str) -> Optional[float]:
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
    # multi-timeframe confirmation for day setups
    day_mtf: dict = field(default_factory=lambda: {
        "enabled": True,
        "confirm_tfs": ["5m","15m","30m","45m"],
        "min_confirmations": 2,
        "require_ema_stack": True,
        "require_macd_bull": True,
        "min_rsi": 52
    })

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
    """
    Keys used (prefix = persistence.key_prefix):
      {p}:state:positions                 -> JSON {'active_positions':{...}}
      {p}:state:performance               -> JSON {'open_trades':[],'closed_trades':[]}
      {p}:state:portfolio                 -> JSON {'cash_usdt':..,'holdings':{sym:{qty,avg}}}
      {p}:silent:open                     -> JSON list of silent positions
      {p}:silent:closed                   -> JSON list of closed silent positions
      {p}:mem:entry_edge:{key}            -> ISO timestamp (TTL)
    """
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url: raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL.")
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48*3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")
    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])
    # state blobs
    def get_json(self, *kparts, default=None):
        txt = self.r.get(self.k(*kparts))
        if not txt: return default
        try: return json.loads(txt)
        except Exception: return default
    def set_json(self, obj, *kparts):
        self.r.set(self.k(*kparts), json.dumps(obj))
    # edge-mem
    def get_mem(self, kind: str, key: str) -> str:
        return self.r.get(self.k("mem", kind, key)) or ""
    def set_mem(self, kind: str, key: str, bar_iso: str):
        self.r.setex(self.k("mem", kind, key), self.ttl_seconds, bar_iso)

# ------------------ stablecoins/pegs ------------------
DEFAULT_STABLES = {
    "USDT","USDC","FDUSD","TUSD","DAI","FRAX","EURS","PAX","PAXG","XAUT","USDJ","USD1","USDD","GUSD",
    "BUSD","USDP","LUSD","PYUSD","WBTC","WETH","stETH","sUSD"
}
def is_stable_symbol(symbol_pair: str, extra_block: Set[str]) -> bool:
    try:
        base, quote = symbol_pair.split("/")
        base = base.upper().replace(".S","")  # normalize
        if base in DEFAULT_STABLES or quote.upper() in DEFAULT_STABLES: return True
        if base in (extra_block or set()): return True
        # also block $1-pegged anomalies by name hints
        peg_hints = ("USD","USDX","XAUT","PAXG","USD1")
        return any(h in base for h in peg_hints)
    except Exception:
        return False

# ================== SD / helpers ==================
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

# ================== Signal engines ==================
def day_signal(df: pd.DataFrame, p: DayParams, sd_cfg: Dict[str,Any], zones=None):
    look, voln = p.lookback_high, p.vol_sma
    if len(df) < max(look, voln)+5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"]> (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else 0)) and (p.rsi_min<=r.iloc[-1]<=p.rsi_max)
    retest_edge = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok   = retest_edge and (last["volume"]>0.8*(volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else 0)) and (r.iloc[-1]>=p.rsi_min)
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
    within=abs((last['close']-last['ema20'])/max(1e-12,last['ema20'])*100)<=p.get('pullback_pct_max',10.0)
    bounce=last['close']>df['close'].iloc[-2]
    hl=df['high'].iloc[-(p.get('breakout_lookback',34)+1):-1].max()
    breakout=(last['close']>hl) and (last['volume']> (df['volS'].iloc[-1] if not np.isnan(df['volS'].iloc[-1]) else 0))
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
    within=abs((last["close"]-last["ema20"])/max(1e-12,last["ema20"])*100)<=p.pullback_pct_max
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(p.breakout_lookback+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]> (df["volS"].iloc[-1] if not np.isnan(df["volS"].iloc[-1]) else 0))
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry=float(last["close"]); stop=stop_from(df,p.stop_mode,p.atr_mult)
    return {"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
            "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout"}

# ---- bearish (for inverse use if needed later; signals feed is bulls only for now) ----
def tf_bull_ok(df: pd.DataFrame, *, require_ema_stack=True, require_macd_bull=True, min_rsi=50) -> bool:
    if df is None or len(df) < 60: return False
    e20 = ema(df['close'], 20); e50 = ema(df['close'], 50); e100 = ema(df['close'], 100)
    macd_line, macd_sig, _ = macd(df['close']); r = rsi(df['close'], 14)
    ok=True
    if require_ema_stack: ok &= (e20.iloc[-1] > e50.iloc[-1] > e100.iloc[-1])
    if require_macd_bull: ok &= (macd_line.iloc[-1] > macd_sig.iloc[-1])
    ok &= (r.iloc[-1] >= min_rsi)
    return bool(ok)

# ================== Movers (legacy logic) ==================
def fetch_cmc_listings(cfg: Dict[str,Any], limit=500) -> List[dict]:
    api_key = os.environ.get("CMC_API_KEY") or cfg.get("movers",{}).get("cmc_api_key")
    if not api_key: return []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": limit, "convert": "USD"}
    r = requests.get(url, headers={"X-CMC_PRO_API_KEY": api_key}, params=params, timeout=20)
    return r.json().get("data", [])

def top100_symbols_on_mexc(client: ExClient, cfg: Dict[str,Any], exclude_stables:Set[str]) -> List[str]:
    data = fetch_cmc_listings(cfg, limit=100)
    out=[]
    for it in data:
        base = it["symbol"].upper()
        pair = f"{base}/USDT"
        if is_stable_symbol(pair, exclude_stables): continue
        if client.has_pair(pair): out.append(pair)
    return out

def movers_legacy_symbols(client: ExClient, cfg: Dict[str,Any], exclude_stables:Set[str]) -> List[str]:
    mv = cfg.get("movers", {})
    if not mv.get("enabled"): return []
    data = fetch_cmc_listings(cfg, limit=mv.get("limit", 500))
    min_change = mv.get("min_change_24h", 15.0)
    min_vol = mv.get("min_volume_usd_24h", 5_000_000)
    max_age = mv.get("max_age_days", 365)
    out=[]
    now=pd.Timestamp.utcnow()
    for it in data:
        sym=it["symbol"].upper()
        q=it.get("quote",{}).get("USD",{})
        ch=(q.get("percent_change_24h") or 0.0)
        vol=(q.get("volume_24h") or 0.0)
        date_added=pd.to_datetime(it.get("date_added", now.isoformat()), utc=True)
        age_days=(now-date_added).days
        if ch>=min_change and vol>=min_vol and age_days<=max_age:
            pair=f"{sym}/USDT"
            if is_stable_symbol(pair, exclude_stables): continue
            if client.has_pair(pair): out.append(pair)
    # optional trend confirm on exchange (fakeout-resistant): 1h and 4h EMA stack + MACD + RSIs
    strong=[]
    for pair in out:
        try:
            df1h = client.ohlcv(pair, "1h", 200)
            df4h = client.ohlcv(pair, "4h", 300)
            ok1 = tf_bull_ok(df1h, require_ema_stack=True, require_macd_bull=True, min_rsi=52)
            ok2 = tf_bull_ok(df4h, require_ema_stack=True, require_macd_bull=True, min_rsi=50)
            if ok1 and ok2:
                strong.append(pair)
        except Exception:
            ...
    return strong

# ================== Silent signals ==================
def silent_key(symbol:str, setup:str)->str:
    return f"{symbol}|{setup}"

def load_silent_open(rds:RedisState)->List[Dict[str,Any]]:
    return rds.get_json("silent","open", default=[]) or []

def save_silent_open(rds:RedisState, arr:List[Dict[str,Any]]):
    rds.set_json(arr, "silent","open")

def load_silent_closed(rds:RedisState)->List[Dict[str,Any]]:
    return rds.get_json("silent","closed", default=[]) or []

def save_silent_closed(rds:RedisState, arr:List[Dict[str,Any]]):
    rds.set_json(arr, "silent","closed")

def _eval_hit_order_bars(rows, entry, stop, t1, t2, tp_priority="target_first"):
    for r in rows:
        hi, lo = r["high"], r["low"]
        hit_stop = (lo <= stop)
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

def fetch_bars_since(client:ExClient, pair:str, tf:str, since_ts:pd.Timestamp, max_lim=400)->List[dict]:
    df = client.ohlcv(pair, tf, max_lim)
    df = df.loc[since_ts:]
    out=[]
    for ts, row in df.iterrows():
        out.append({"ts": ts, "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"])})
    return out

def evaluate_silent_signals(rds:RedisState, client:ExClient, cfg:Dict[str,Any]):
    """Close silent signals on t1 or stop; append to closed list with outcome."""
    openS = load_silent_open(rds)
    closedS = load_silent_closed(rds)
    if not openS: return
    tp_prio = cfg.get("performance",{}).get("tp_priority","target_first")
    max_eval = cfg.get("performance",{}).get("silent_max_bars", {"day":240,"swing":200,"trend":180})
    keep=[]
    for s in openS:
        try:
            pair = s["symbol"]; setup = s["setup"]; tf = s["tf"]; entry=s["entry"]; stop=s["stop"]; t1=s.get("t1"); t2=s.get("t2")
            since = pd.to_datetime(s["opened_at"], utc=True)  # already ISO
            rows = fetch_bars_since(client, pair, tf, since)
            if len(rows)<=1:
                keep.append(s); continue
            eval_rows = rows[1:]  # start after signal bar close
            # cap
            tf_key = {"1h":"day","4h":"swing","1d":"trend"}.get(tf, setup)
            max_n = int(max_eval.get(tf_key, 180))
            if max_n and len(eval_rows)>max_n:
                eval_rows = eval_rows[:max_n]
            outcome, price, ts_hit = _eval_hit_order_bars(eval_rows, entry, stop, t1, t2, tp_priority=tp_prio)
            if outcome:
                s["status"]="closed"; s["outcome"]=outcome; s["exit_price"]=float(price); s["closed_at"]=str(ts_hit)
                # r-multiple
                risk = max(1e-12, entry-stop)
                s["r_multiple"]=float((price-entry)/risk)
                s["pct_return"]=float((price/entry-1.0)*100.0)
                closedS.append(s)
            else:
                keep.append(s)
        except Exception as e:
            keep.append(s)
    save_silent_open(rds, keep)
    save_silent_closed(rds, closedS)

def open_silent_if_new(rds:RedisState, symbol:str, setup:str, tf:str, entry:float, stop:float, t1:float, t2:float, opened_at_iso:str):
    arr = load_silent_open(rds)
    k = silent_key(symbol, setup)
    exists = any((x.get("key")==k and x.get("status","open")=="open") for x in arr)
    if exists: return False
    arr.append({
        "key": k, "symbol": symbol, "setup": setup, "tf": tf,
        "entry": float(entry), "stop": float(stop), "t1": float(t1) if t1 else None, "t2": float(t2) if t2 else None,
        "opened_at": opened_at_iso, "status":"open"
    })
    save_silent_open(rds, arr)
    return True

def silent_perf_dashboard(rds:RedisState):
    openS = load_silent_open(rds)
    closedS = load_silent_closed(rds)
    def filt(grp): return [x for x in closedS if x.get("setup")==grp]
    print("\n--- Silent Signal Performance (since reset) ---")
    for grp in ["day","swing","trend","mover"]:
        cl = filt(grp)
        op = [x for x in openS if x.get("setup")==grp]
        if cl:
            df = pd.DataFrame(cl)
            win = float((df["r_multiple"]>0).mean()*100.0) if "r_multiple" in df else 0.0
            avgR = float(df["r_multiple"].mean()) if "r_multiple" in df else 0.0
            bestR = float(df["r_multiple"].max()) if "r_multiple" in df else 0.0
            worstR= float(df["r_multiple"].min()) if "r_multiple" in df else 0.0
            print(f"{grp.capitalize():<6}: closed {len(cl)} | open {len(op)} | Win% {win:.1f}% | AvgR {avgR:.2f} | Best/Worst {bestR:.2f}/{worstR:.2f}")
        else:
            print(f"{grp.capitalize():<6}: closed 0 | open {len(op)}")
    print("--- End Silent Performance ---\n")

# ================== Positions / Trades state ==================
def load_positions(rds:RedisState)->Dict[str,Any]:
    return rds.get_json("state","positions", default={"active_positions":{}}) or {"active_positions":{}}
def save_positions(rds:RedisState, d:Dict[str,Any]): rds.set_json(d, "state","positions")
def load_perf(rds:RedisState)->Dict[str,Any]:
    return rds.get_json("state","performance", default={"open_trades":[], "closed_trades":[]}) or {"open_trades":[], "closed_trades":[]}
def save_perf(rds:RedisState, d:Dict[str,Any]): rds.set_json(d, "state","performance")
def load_portfolio(rds:RedisState)->Dict[str,Any]:
    return rds.get_json("state","portfolio", default={}) or {}
def save_portfolio(rds:RedisState, d:Dict[str,Any]): rds.set_json(d, "state","portfolio")

# ================== Paper portfolio & trade mgmt ==================
def ensure_portfolio(rds: RedisState, trading_cfg: Dict[str,Any]) -> Dict[str,Any]:
    pf = load_portfolio(rds)
    if not pf:
        base_bal = float(trading_cfg.get("base_balance_usdt", 1000.0))
        pf = {"cash_usdt": base_bal, "holdings": {}}
        save_portfolio(rds, pf)
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
    save_portfolio(rds, pf)
    return True

def paper_sell(rds: RedisState, pf: Dict[str,Any], symbol: str, price: float, qty: float) -> float:
    h = pf["holdings"].get(symbol)
    if not h or qty<=0: return 0.0
    sell_qty = min(qty, h["qty"])
    proceeds = price * sell_qty
    pf["cash_usdt"] += proceeds
    h["qty"] -= sell_qty
    if h["qty"] <= 1e-12:
        pf["holdings"].pop(symbol, None)
    save_portfolio(rds, pf)
    return proceeds

def paper_snapshot(client: ExClient, rds: RedisState, pf: Dict[str,Any], perf: Dict[str,Any]):
    mv_total = 0.0; pnl_total = 0.0; lines=[]
    for sym, pos in sorted(pf.get("holdings", {}).items()):
        qty = float(pos["qty"]); avg = float(pos["avg"])
        px = client.ticker_last(sym) or avg
        mv = qty * px
        pnl = (px - avg) * qty
        mv_total += mv; pnl_total += pnl
        lines.append((sym, qty, avg, px, pnl, mv))
    equity = pf.get("cash_usdt", 0.0) + mv_total
    exposure = mv_total
    open_R = 0.0
    for tr in perf.get("open_trades", []):
        risk = float(tr.get("risk", 0.0)) or 1e-12
        entry = float(tr.get("entry", 0.0))
        px = client.ticker_last(tr["symbol"]) or entry
        open_R += (px - entry) / risk
    closed = perf.get("closed_trades", [])
    win, avgR, medR, bestR, worstR, pfactor = 0.0,0.0,0.0,0.0,0.0,0.0
    if closed:
        dfc = pd.DataFrame(closed)
        if "r_multiple" in dfc.columns:
            win = float((dfc["r_multiple"]>0).mean()*100.0)
            avgR = float(dfc["r_multiple"].mean()); medR = float(dfc["r_multiple"].median())
            bestR= float(dfc["r_multiple"].max());  worstR= float(dfc["r_multiple"].min())
            gains = dfc.loc[dfc["r_multiple"]>0,"r_multiple"].sum()
            losses= -dfc.loc[dfc["r_multiple"]<0,"r_multiple"].sum()
            pfactor = float(gains/losses) if losses>0 else float("inf")
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
    print("--- End Snapshot ---\n")

# ================== Trades perf ==================
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
        "filled_qty": 0.0
    })

def close_trade_record(perf:Dict[str,Any], tr:dict, *, outcome:str, px:float, when_iso:str, reason:str):
    tr["status"]="closed"; tr["closed_at"]=when_iso; tr["exit_price"]=float(px); tr["outcome"]=outcome; tr["reason"]=reason
    tr["r_multiple"]=float((px - tr["entry"]) / max(1e-12, tr["risk"]))
    tr["pct_return"]=float((px/tr["entry"] - 1.0)*100.0)
    perf.setdefault("closed_trades", []).append(tr)

# ================== Discord ==================
def post_discord(webhook:str, content:str):
    if not webhook: return
    try:
        requests.post(webhook, json={"content": content}, timeout=10)
    except Exception as e:
        print("[discord] post error:", e)

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

    exclude_stables = set(cfg.get("filters", {}).get("extra_stables", []))

    # Params
    dayP   = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = TrendParams(**cfg.get("trend_trade_params", {}))
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    qcfg   = cfg.get("quality", {})
    movers_cfg = cfg.get("movers", {"enabled": True})

    # Discord webhooks
    signals_wh = os.environ.get("DISCORD_SIGNALS_WEBHOOK", cfg.get("discord",{}).get("signals_webhook",""))
    trades_wh  = os.environ.get("DISCORD_TRADES_WEBHOOK",  cfg.get("discord",{}).get("trades_webhook",""))

    # ---- Build scan lists
    scan_top100 = top100_symbols_on_mexc(client, cfg, exclude_stables)
    scan_movers = movers_legacy_symbols(client, cfg, exclude_stables)

    # SD zones cache
    zones_cache={}
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        uniq_pairs = list(dict.fromkeys(scan_top100 + scan_movers))
        for pair in uniq_pairs:
            try:
                zdf = client.ohlcv(pair, ztf, zlook)
                zones_cache[pair] = find_zones(
                    zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                )
            except Exception as e:
                print(f"[zones] {pair} err:", e); zones_cache[pair]=[]

    positions = load_positions(rds)
    perf      = load_perf(rds)
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}

    # ================== Generate signals ==================
    signals_top=[]; signals_movers=[]

    # helper: apply MTF confirmation for day signals
    def mtf_confirm_day(pair:str, sig:dict)->bool:
        mtf = dayP.day_mtf or {}
        if not mtf.get("enabled", True): return True
        tfs = mtf.get("confirm_tfs", ["5m","15m","30m","45m"])
        need = int(mtf.get("min_confirmations", 2))
        hits=0
        for tf in tfs:
            try:
                df = client.ohlcv(pair, tf, 200)
                if tf_bull_ok(df,
                    require_ema_stack=mtf.get("require_ema_stack", True),
                    require_macd_bull=mtf.get("require_macd_bull", True),
                    min_rsi=mtf.get("min_rsi", 52)):
                    hits += 1
            except Exception:
                ...
        return hits >= need

    # ---- Top100 scan
    for pair in scan_top100:
        try:
            # DAY (1h)
            df1h = client.ohlcv(pair, "1h", 300); zones=zones_cache.get(pair)
            dsig = day_signal(df1h, dayP, sd_cfg, zones)
            if dsig and mtf_confirm_day(pair, dsig):
                dsig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc","origin":"top100"})
                signals_top.append(dsig)
            # SWING (4h)
            tf_swing = swingP.get("timeframe","4h")
            df4h = client.ohlcv(pair, tf_swing, 400); zones=zones_cache.get(pair)
            ssig = swing_signal(df4h, swingP, sd_cfg, zones)
            if ssig:
                ssig.update({"symbol":pair,"timeframe":tf_swing,"exchange":"mexc","origin":"top100"})
                signals_top.append(ssig)
            # TREND (1d)
            dfd = client.ohlcv(pair, "1d", 320); zones=zones_cache.get(pair)
            tsig = trend_signal(dfd, trnP, sd_cfg, zones)
            if tsig:
                tsig.update({"symbol":pair,"timeframe":"1d","exchange":"mexc","origin":"top100"})
                signals_top.append(tsig)
        except Exception as e:
            print(f"[scan-top100] {pair} err:", e)

    # ---- Movers scan (legacy logic; 1h only, labeled 'mover')
    for pair in scan_movers:
        try:
            df1h = client.ohlcv(pair, "1h", 300); zones=zones_cache.get(pair)
            # stronger thresholds for movers → reuse day_signal but label and require SD optional
            sig = day_signal(df1h, dayP, sd_cfg, zones)
            if sig:
                # tag as mover, slightly raise targets for momentum
                sig["type"]="day"; sig["note"]="Mover Trend"
                sig["t1"]=round(sig["entry"]*1.05,6); sig["t2"]=round(sig["entry"]*1.10,6)
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc","origin":"movers"})
                # quick extra confirmation: last 10 bars higher lows / ema20 rising
                e20 = ema(df1h["close"],20)
                if e20.iloc[-1] > e20.iloc[-5]:
                    signals_movers.append(sig)
        except Exception as e:
            print(f"[scan-movers] {pair} err:", e)

    # ================== Deduplicate by silent signals (no dup until closed) ==================
    def allowed_by_silent(sig)->bool:
        setup = sig["type"] if sig.get("origin")!="movers" else "mover"
        k = silent_key(sig["symbol"], setup)
        openS = load_silent_open(rds)
        return not any((x.get("key")==k and x.get("status","open")=="open") for x in openS)

    new_signals_top = [s for s in signals_top if allowed_by_silent(s)]
    new_signals_mov = [s for s in signals_movers if allowed_by_silent(s)]

    # open silent records
    for s in (new_signals_top + new_signals_mov):
        setup = s["type"] if s.get("origin")!="movers" else "mover"
        opened_at_iso = s.get("event_bar_ts") or pd.Timestamp.utcnow().isoformat()
        open_silent_if_new(rds, s["symbol"], setup, s["timeframe"], s["entry"], s["stop"], s.get("t1"), s.get("t2"), opened_at_iso)

    # ================== Execute real trades (respect max positions)
    active_syms = set(v["symbol"] for v in positions.get("active_positions", {}).values())
    n_open = len(active_syms)
    executed_trades=[]

    def try_execute(sig:dict):
        nonlocal n_open
        sym = sig["symbol"]; tf=sig["timeframe"]; typ=sig["type"]; entry=sig["entry"]; stop=sig["stop"]
        if sym in active_syms: return
        if n_open >= max_pos: return
        # sizing
        equity = (load_portfolio(rds).get("cash_usdt",0.0) if paper_mode else float(client.ex.fetch_balance().get("USDT",{}).get("free",0.0)))
        qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
        notional = qty * entry
        if notional < min_order or (not paper_mode and notional>equity):
            return
        # execute
        if paper_mode:
            pf_local = load_portfolio(rds)
            if not paper_buy(rds, pf_local, sym, entry, qty, min_order): return
            executed_trades.append(f"BUY `{sym}` {tf} qty `{qty:.6f}` @ `{entry:.6f}` (paper)")
        else:
            try:
                client.place_market(sym, "buy", qty)
                executed_trades.append(f"BUY `{sym}` {tf} qty `{qty:.6f}` @ `~{entry*(1+slip_bps/10000):.6f}` (live)")
            except Exception as e:
                print("[live] order err", sym, e); return
        # register pos & trade perf
        positions["active_positions"][f"mexc|{sym}|{typ}"] = {"exchange":"mexc","symbol":sym,"type":typ,"entry":entry,"timeframe":tf,"ts":pd.Timestamp.utcnow().isoformat()}
        save_positions(rds, positions)
        add_open_trade(load_perf(rds), exchange="mexc", symbol=sym, tf=tf, sig_type=typ, entry=entry, stop=stop,
                       t1=sig.get("t1"), t2=sig.get("t2"),
                       event_ts=(s.get("event_bar_ts") or pd.Timestamp.utcnow().isoformat()))
        save_perf(rds, load_perf(rds))
        active_syms.add(sym); n_open += 1

    # Execute from Top100 then Movers (priority)
    for s in new_signals_top: try_execute(s)
    for s in new_signals_mov: try_execute(s)

    # ================== Evaluate silent signals (t1/stop close)
    evaluate_silent_signals(rds, client, cfg)

    # ================== Logs & Discord ==================
    now_iso = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== MEXC Signals @ {pd.Timestamp.utcnow().isoformat()} ===")
    print(f"Scanned — 1h:{len(scan_top100)}  4h:{len(scan_top100)}  1d:{len(scan_top100)}  | movers:{len(scan_movers)}")
    if scan_movers:
        print("Movers (MEXC):", ", ".join(scan_movers[:20]) + ("..." if len(scan_movers)>20 else ""))

    # SIGNALS — send only signal info, no skip/executed labels
    if (new_signals_top or new_signals_mov) and signals_wh:
        parts=[]
        if new_signals_top:
            parts.append("**Signals — Top 100**")
            for s in new_signals_top:
                parts.append(f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`")
        if new_signals_mov:
            parts.append("**Signals — Movers**")
            for s in new_signals_mov:
                parts.append(f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`")
        post_discord(signals_wh, "\n".join(parts))

    # TRADES — only post when there were executions this run
    if executed_trades and trades_wh:
        post_discord(trades_wh, "💼 **Trades @ "+now_iso+"**\n" + "\n".join("• "+x for x in executed_trades))

    # Silent performance (console only)
    silent_perf_dashboard(rds)

    # Paper portfolio snapshot (console)
    if paper_mode:
        paper_snapshot(client, rds, load_portfolio(rds), load_perf(rds))


# ================== Entrypoint ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
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
