#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mexc_trader_bot.py — MEXC spot scanner/trader with:
- Top-100 (CMC) signal scan (replaces legacy watchlist for signals)
- Separate legacy Movers scan (15%+, volume, trend-confirm, fakeout-resistant)
- Redis-backed *edge de-dup per bar* + *cooldown bars* across ALL signal paths
- Paper/live trading, max positions, performance snapshot
- Discord: separate webhooks (signals vs trades). Signals always listed (no "skipped/executed"), Trades only when an order executes
- Stablecoins are excluded from Movers

ENV expected:
  REDIS_URL
  CMC_API_KEY
  DISCORD_SIGNALS_WEBHOOK
  DISCORD_TRADES_WEBHOOK
  (optional) MEXC_API_KEY, MEXC_SECRET
"""

import argparse, json, os, sys, yaml, requests, math
from typing import Dict, Any, List, Tuple, Optional
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
    # --- edge memories (entries) ---
    def get_entry_edge(self, key: str) -> str:
        return self.r.get(self.k("mem","entry_edge", key)) or ""
    def set_entry_edge(self, key: str, bar_iso: str, ttl_sec: int = None):
        ttl = ttl_sec or self.ttl_seconds
        self.r.setex(self.k("mem","entry_edge", key), ttl, bar_iso)
    # --- last signal time for cooldown ---
    def get_last_sig_ts(self, key: str) -> str:
        return self.r.get(self.k("mem","last_sig", key)) or ""
    def set_last_sig_ts(self, key: str, bar_iso: str, ttl_sec: int = None):
        ttl = ttl_sec or self.ttl_seconds
        self.r.setex(self.k("mem","last_sig", key), ttl, bar_iso)
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
def day_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    look = int(p.get("lookback_high", 30)); voln = int(p.get("vol_sma", 30))
    if len(df) < max(look, voln)+5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"]>volS.iloc[-1]) and (p.get("rsi_min",52)<=r.iloc[-1]<=p.get("rsi_max",78))
    retest_edge = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok   = retest_edge and (last["volume"]>0.8*volS.iloc[-1]) and (r.iloc[-1]>=p.get("rsi_min",52))
    if not (breakout_ok or retrec_ok): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry = float(last["close"]); stop = stop_from(df, p.get("stop_mode","swing"), float(p.get("atr_mult",1.5)))
    return {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
            "level":float(highlvl),"note":"Breakout" if breakout_ok else "Retest-Reclaim"}

def swing_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    need = max(int(p.get('ema100',100)), int(p.get('vol_sma',20)), int(p.get('breakout_lookback',34)))+5
    if len(df)<need: return None
    df=df.copy(); df['ema20']=ema(df['close'],int(p.get('ema20',20))); df['ema50']=ema(df['close'],int(p.get('ema50',50)))
    df['ema100']=ema(df['close'],int(p.get('ema100',100))); df['volS']=sma(df['volume'],int(p.get('vol_sma',20)))
    r=rsi(df['close'],14); last=df.iloc[-1]
    aligned=(last['ema20']>last['ema50']>last['ema100']) and (r.iloc[-1]>=int(p.get('rsi_min',50)))
    within=abs((last['close']-last['ema20'])/max(1e-12,last['ema20'])*100)<=float(p.get('pullback_pct_max',10.0))
    bounce=last['close']>df['close'].iloc[-2]
    hl=df['high'].iloc[-(int(p.get('breakout_lookback',34))+1):-1].max()
    breakout=(last['close']>hl) and (last['volume']>df['volS'].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get('enabled') and sd_cfg.get('mode','prefer')=='require' and not in_demand(float(last['close']), zones): return None
    entry=float(last['close']); stop=stop_from(df,p.get('stop_mode','swing'),float(p.get('atr_mult',2.0)))
    return {"type":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
            "level":float(hl),"note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout"}

def trend_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    need=max(int(p.get("ema100",100)), int(p.get("vol_sma",20)), int(p.get("breakout_lookback",55)))+5
    if len(df)<need: return None
    df=df.copy(); df["ema20"]=ema(df["close"],int(p.get("ema20",20))); df["ema50"]=ema(df["close"],int(p.get("ema50",50)))
    df["ema100"]=ema(df["close"],int(p.get("ema100",100))); df["volS"]=sma(df["volume"],int(p.get("vol_sma",20)))
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1]>=int(p.get("rsi_min",50)))
    within=abs((last["close"]-last["ema20"])/max(1e-12,last["ema20"])*100)<=float(p.get("pullback_pct_max",10.0))
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(int(p.get("breakout_lookback",55))+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]>df["volS"].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry=float(last["close"]); stop=stop_from(df,p.get("stop_mode","swing"),float(p.get("atr_mult",2.0)))
    return {"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
            "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout"}

# ================== Legacy Movers (15%+, big vol, trend confirm, fakeout-resistant) ==================
STABLECOIN_BASES = {"USDT","USDC","FDUSD","TUSD","DAI","USD","USD1","USDD","USDP","PAXG","XAUT"}  # last two are gold proxies; exclude from Movers per your request

def fetch_cmc_list(cfg: Dict[str,Any], limit: int = 500) -> List[dict]:
    api_key = os.environ.get("CMC_API_KEY") or (cfg.get("movers", {}) or {}).get("cmc_api_key")
    if not api_key:
        print("[cmc] no CMC_API_KEY; returning []")
        return []
    headers = {"X-CMC_PRO_API_KEY": api_key}
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": limit, "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print("[cmc] error:", e)
        return []

def top100_symbols_on_mexc(client: ExClient, cfg: Dict[str,Any]) -> List[str]:
    data = fetch_cmc_list(cfg, limit=120)
    # take top-100 by market cap
    data = sorted(data, key=lambda x: x.get("cmc_rank", 999999))[:100]
    syms = [d["symbol"].upper() for d in data]
    # map to /USDT pairs present in MEXC
    out = []
    client.load_markets()
    for s in syms:
        pair = f"{s}/USDT"
        if client.has_pair(pair):
            out.append(pair)
    return out

def legacy_movers_symbols_on_mexc(client: ExClient, cfg: Dict[str,Any]) -> List[str]:
    mv = cfg.get("movers", {}) or {}
    min_change = float(mv.get("min_change_24h", 15.0))
    min_vol    = float(mv.get("min_volume_usd_24h", 5_000_000))
    max_age    = int(mv.get("max_age_days", 365))
    data = fetch_cmc_list(cfg, limit=int(mv.get("limit", 500)))
    now = pd.Timestamp.utcnow()
    picks = []
    for it in data:
        sym = it["symbol"].upper()
        if sym in STABLECOIN_BASES:
            continue
        q = (it.get("quote",{}) or {}).get("USD",{}) or {}
        ch = float(q.get("percent_change_24h") or 0.0)
        vol = float(q.get("volume_24h") or 0.0)
        date_added = pd.to_datetime(it.get("date_added", now.isoformat()), utc=True)
        age_days = (now - date_added).days
        if (ch >= min_change) and (vol >= min_vol) and (age_days <= max_age):
            pair = f"{sym}/USDT"
            if client.has_pair(pair):
                picks.append(pair)
    return picks

def confirm_strong_trend(df: pd.DataFrame) -> bool:
    """Fakeout-resistant confirmation:
       - EMA20 > EMA50 (bull stack)
       - RSI(14) >= 55
       - MACD line > signal
       - Close near high of bar (upper 40%) to avoid long upper-wick fakeouts
    """
    if df is None or len(df) < 60: return False
    e20, e50 = ema(df["close"], 20), ema(df["close"], 50)
    r = rsi(df["close"], 14)
    line, sig, _ = macd(df["close"])
    last = df.iloc[-1]
    rng = max(1e-12, last["high"] - last["low"])
    close_pos = (last["close"] - last["low"]) / rng  # 0..1
    return (e20.iloc[-1] > e50.iloc[-1]) and (r.iloc[-1] >= 55) and (line.iloc[-1] > sig.iloc[-1]) and (close_pos >= 0.6)

# ================== BTC filter ==================
def btc_ok(client: ExClient, p: Dict[str,Any]) -> bool:
    if not p.get("btc_filter", False):
        return True
    try:
        df = client.ohlcv(p.get("btc_symbol","BTC/USDT"), "1h", 120)
        return df["close"].iloc[-1] > ema(df["close"], int(p.get("btc_ema", 20))).iloc[-1]
    except Exception:
        return True

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

# ================== Paper performance snapshot ==================
def safe_col(df: pd.DataFrame, col: str, default_val=None):
    return df[col] if (isinstance(df, pd.DataFrame) and col in df.columns) else default_val

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

    # Open R (approx)
    open_R = 0.0
    for tr in perf.get("open_trades", []):
        risk = float(tr.get("risk", 0.0)) or 1e-12
        entry = float(tr.get("entry", 0.0))
        px = client.last_price(tr.get("symbol","")) or entry
        open_R += (px - entry) / risk

    # Closed stats (robust to missing columns)
    closed = perf.get("closed_trades", [])
    win=avgR=medR=bestR=worstR=pfactor=0.0
    if closed:
        dfc = pd.DataFrame(closed)
        if "r_multiple" in dfc.columns:
            win = float((safe_col(dfc, "r_multiple", pd.Series([])) > 0).mean() * 100.0)
            rms = safe_col(dfc, "r_multiple", pd.Series([0.0]))
            avgR = float(rms.mean()) if len(rms) else 0.0
            medR = float(rms.median()) if len(rms) else 0.0
            bestR = float(rms.max()) if len(rms) else 0.0
            worstR = float(rms.min()) if len(rms) else 0.0
            gains = rms[rms>0].sum()
            losses = -rms[rms<0].sum()
            pfactor = float(gains / losses) if losses > 0 else float("inf")
        elif "outcome" in dfc.columns:
            win = float((dfc["outcome"].isin(["t1","t2"])).mean() * 100.0)

    print("\n--- Paper Performance Snapshot ---")
    print(f"Cash:      {pf.get('cash_usdt',0.0):.2f} USDT")
    print(f"Exposure:  {exposure:.2f} USDT  | Positions: {len(pf.get('holdings',{}))}")
    print(f"Equity:    {equity:.2f} USDT  | Unrealized PnL: {pnl_total:+.2f} USDT")
    print(f"Open R:    {open_R:+.2f} R (sum)")
    if closed:
        print(f"Closed n:  {len(closed)} | Win%: {win:.1f}% | AvgR: {avgR:.2f} | MedR: {medR:.2f} | PF: {pfactor if pfactor!=float('inf') else 'inf'} | Best/Worst R: {bestR:.2f}/{worstR:.2f}")
    print("--- End Snapshot ---\n")

# ================== Cooldown / edge de-dup ==================
TF_MINUTES = {"5m":5, "15m":15, "30m":30, "1h":60, "2h":120, "4h":240, "1d":1440}
def bars_since(prev_iso: str, now_iso: str, tf: str) -> float:
    if not prev_iso: return float('inf')
    try:
        prev = pd.to_datetime(prev_iso, utc=True)
        now  = pd.to_datetime(now_iso,  utc=True)
    except Exception:
        return float('inf')
    mins = (now - prev).total_seconds() / 60.0
    tfmin = TF_MINUTES.get(tf, 60)
    return mins / max(tfmin,1)

def cooldown_ok_redis(rds: RedisState, *, ex: str, sym: str, tf: str, typ: str, bars_needed: int, bar_iso: str) -> bool:
    k = f"{ex}|{sym}|{tf}|{typ}"
    last = rds.get_last_sig_ts(k)
    return bars_since(last, bar_iso, tf) >= bars_needed

# ================== Discord ==================
def post_discord(webhook: str, content: str):
    if not webhook or not content.strip():
        return
    try:
        requests.post(webhook, json={"content": content}, timeout=10)
    except Exception as e:
        print("[discord] err:", e)

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

    sd_cfg    = cfg.get("supply_demand", {"enabled": False})
    dayP      = cfg.get("day_trade_params", {}) or {}
    swingP    = cfg.get("swing_trade_params", {"timeframe":"4h"}) or {"timeframe":"4h"}
    trnP      = cfg.get("trend_trade_params", {}) or {}
    qcfg      = cfg.get("quality", {}) or {}
    movers_cfg= cfg.get("movers", {}) or {}

    # Load state
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}

    # Build scans
    top100_pairs = top100_symbols_on_mexc(client, cfg)              # used for normal signal scan
    movers_pairs = legacy_movers_symbols_on_mexc(client, cfg)       # separate movers scan (legacy logic)

    def uniq(seq): return list(dict.fromkeys(seq))
    scan_pairs_day_top   = uniq(top100_pairs)
    scan_pairs_swing_top = uniq(top100_pairs)
    scan_pairs_trend_top = [p for p in top100_pairs if p not in []]  # same set; can restrict if wanted

    # SD zones cache
    zones_cache = {}
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        for pair in uniq(scan_pairs_day_top + scan_pairs_swing_top + scan_pairs_trend_top + movers_pairs):
            try:
                zdf = client.ohlcv(pair, ztf, zlook)
                zones_cache[pair] = find_zones(
                    zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                )
            except Exception as e:
                print(f"[zones] {pair} err:", e); zones_cache[pair]=[]

    results = {"ts": pd.Timestamp.utcnow().isoformat(), "signals_top": [], "signals_movers": []}

    # ===== Top-100 scans =====
    allow_day = btc_ok(client, dayP)
    # DAY (1h)
    for pair in scan_pairs_day_top:
        try:
            df1h = client.ohlcv(pair, "1h", 300); zones = zones_cache.get(pair)
            sig = day_signal(df1h, dayP, sd_cfg, zones)
            if sig and allow_day:
                tf = "1h"; ex_name = "mexc"; event_bar = df1h.index[-1].isoformat()
                # edge de-dup + cooldown
                edge_key = f"{ex_name}|{pair}|{tf}|{sig['type']}|{sig.get('note','')}"
                last_edge = rds.get_entry_edge(edge_key)
                if last_edge != event_bar:
                    bars_needed = int(qcfg.get("signal_cooldown_bars", 6))
                    if cooldown_ok_redis(rds, ex=ex_name, sym=pair, tf=tf, typ=sig["type"], bars_needed=bars_needed, bar_iso=event_bar):
                        rds.set_entry_edge(edge_key, event_bar)
                        rds.set_last_sig_ts(f"{ex_name}|{pair}|{tf}|{sig['type']}", event_bar)
                        sig.update({"symbol":pair,"timeframe":tf,"exchange":ex_name,"event_bar_ts":event_bar})
                        results["signals_top"].append(sig)
        except Exception as e: print(f"[scan-day-top] {pair} err:", e)

    # SWING (4h)
    tf_swing = swingP.get("timeframe","4h")
    for pair in scan_pairs_swing_top:
        try:
            df4h = client.ohlcv(pair, tf_swing, 400); zones = zones_cache.get(pair)
            sig = swing_signal(df4h, swingP, sd_cfg, zones)
            if sig:
                tf = tf_swing; ex_name = "mexc"; event_bar = df4h.index[-1].isoformat()
                edge_key = f"{ex_name}|{pair}|{tf}|{sig['type']}|{sig.get('note','')}"
                last_edge = rds.get_entry_edge(edge_key)
                if last_edge != event_bar:
                    bars_needed = int(qcfg.get("signal_cooldown_bars", 6))
                    if cooldown_ok_redis(rds, ex=ex_name, sym=pair, tf=tf, typ=sig["type"], bars_needed=bars_needed, bar_iso=event_bar):
                        rds.set_entry_edge(edge_key, event_bar)
                        rds.set_last_sig_ts(f"{ex_name}|{pair}|{tf}|{sig['type']}", event_bar)
                        sig.update({"symbol":pair,"timeframe":tf,"exchange":"mexc","event_bar_ts":event_bar})
                        results["signals_top"].append(sig)
        except Exception as e: print(f"[scan-swing-top] {pair} err:", e)

    # TREND (1d)
    for pair in scan_pairs_trend_top:
        try:
            dfd = client.ohlcv(pair, "1d", 320); zones = zones_cache.get(pair)
            sig = trend_signal(dfd, trnP, sd_cfg, zones)
            if sig:
                tf = "1d"; ex_name = "mexc"; event_bar = dfd.index[-1].isoformat()
                edge_key = f"{ex_name}|{pair}|{tf}|{sig['type']}|{sig.get('note','')}"
                last_edge = rds.get_entry_edge(edge_key)
                if last_edge != event_bar:
                    bars_needed = int(qcfg.get("signal_cooldown_bars", 6))
                    if cooldown_ok_redis(rds, ex=ex_name, sym=pair, tf=tf, typ=sig["type"], bars_needed=bars_needed, bar_iso=event_bar):
                        rds.set_entry_edge(edge_key, event_bar)
                        rds.set_last_sig_ts(f"{ex_name}|{pair}|{tf}|{sig['type']}", event_bar)
                        sig.update({"symbol":pair,"timeframe":tf,"exchange":"mexc","event_bar_ts":event_bar})
                        results["signals_top"].append(sig)
        except Exception as e: print(f"[scan-trend-top] {pair} err:", e)

    # ===== Movers scan (legacy logic + trend confirm + no stablecoins) =====
    for pair in movers_pairs:
        try:
            # Use 1h TF for movers entries
            df1h = client.ohlcv(pair, "1h", 300)
            if not confirm_strong_trend(df1h):
                continue
            # Reuse day_signal thresholds (or could tailor movers-specific targets)
            sig = day_signal(df1h, dayP, sd_cfg, zones_cache.get(pair))
            if sig:
                sig["note"] = "Mover Trend"
                tf = "1h"; ex_name="mexc"; event_bar = df1h.index[-1].isoformat()
                edge_key = f"{ex_name}|{pair}|{tf}|{sig['type']}|{sig.get('note','')}"
                last_edge = rds.get_entry_edge(edge_key)
                if last_edge != event_bar:
                    bars_needed = int(qcfg.get("signal_cooldown_bars", 6))
                    if cooldown_ok_redis(rds, ex=ex_name, sym=pair, tf=tf, typ=sig["type"], bars_needed=bars_needed, bar_iso=event_bar):
                        rds.set_entry_edge(edge_key, event_bar)
                        rds.set_last_sig_ts(f"{ex_name}|{pair}|{tf}|{sig['type']}", event_bar)
                        sig.update({"symbol":pair,"timeframe":tf,"exchange":"mexc","event_bar_ts":event_bar})
                        results["signals_movers"].append(sig)
        except Exception as e:
            print(f"[scan-movers] {pair} err:", e)

    # ===== EXECUTE entries (paper or live) =====
    already_open_symbols = set()
    for k, v in positions.get("active_positions", {}).items():
        already_open_symbols.add(v["symbol"])
    n_open = len(already_open_symbols)

    actionable_signals = []  # for execution decisions
    for sig in (results["signals_top"] + results["signals_movers"]):
        sym = sig["symbol"]; tf = sig["timeframe"]; typ = sig["type"]; entry = sig["entry"]; stop = sig["stop"]
        # respect max positions; but regardless of trade, the signal will be discord-logged in signals channel later
        if sym in already_open_symbols or n_open >= max_pos:
            continue
        actionable_signals.append(sig)
        # sizing/execution
        if paper_mode:
            equity = pf.get("cash_usdt", 0.0)
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if not ok:
                continue
            # record position and perf
            update_positions_active(positions, {"exchange":"mexc","symbol":sym,"type":typ,"entry":entry,"timeframe":tf})
            add_open_trade(perf, exchange="mexc", symbol=sym, tf=tf, sig_type=typ,
                           entry=entry, stop=stop, t1=sig.get("t1"), t2=sig.get("t2"),
                           event_ts=sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()))
            already_open_symbols.add(sym); n_open += 1
            # Discord TRADES (only when there is a trade)
            trades_wh = os.environ.get("DISCORD_TRADES_WEBHOOK", "")
            post_discord(trades_wh, f"**Trades @ {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC**\n"
                                    f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `{entry:.6f}` (paper)")
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
                update_positions_active(positions, {"exchange":"mexc","symbol":sym,"type":typ,"entry":px,"timeframe":tf})
                add_open_trade(perf, exchange="mexc", symbol=sym, tf=tf, sig_type=typ,
                               entry=px, stop=stop, t1=sig.get("t1"), t2=sig.get("t2"),
                               event_ts=sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()))
                already_open_symbols.add(sym); n_open += 1
                trades_wh = os.environ.get("DISCORD_TRADES_WEBHOOK", "")
                post_discord(trades_wh, f"**Trades @ {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC**\n"
                                        f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `~{px:.6f}` (live)")
            except Exception as e:
                print(f"[live] order err {sym}:", e); continue

    # Save states
    rds.save_positions(positions)
    rds.save_perf(perf)

    # ===== Logs =====
    print(f"=== MEXC Signals @ {results['ts']} ===")
    print(f"Scanned — 1h:{len(scan_pairs_day_top)}  4h:{len(scan_pairs_swing_top)}  1d:{len(scan_pairs_trend_top)}  | top100:{len(top100_pairs)}  movers:{len(movers_pairs)}")
    if movers_pairs:
        print("Movers (MEXC):", ", ".join(movers_pairs))

    # Discord — Signals (two blocks: Top 100 and Movers). Only send if non-empty.
    sig_wh = os.environ.get("DISCORD_SIGNALS_WEBHOOK", "")
    if results["signals_top"]:
        parts = ["**Signals — Top 100**"]
        for s in results["signals_top"]:
            parts.append(f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n"
                         f"  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`")
        post_discord(sig_wh, "\n".join(parts))
    if results["signals_movers"]:
        parts = ["**Signals — Movers**"]
        for s in results["signals_movers"]:
            parts.append(f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n"
                         f"  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`")
        post_discord(sig_wh, "\n".join(parts))

    # Paper snapshot (every run)
    if paper_mode:
        paper_snapshot(client, rds, pf, perf)

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
