#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mexc_trader_bot.py — MEXC spot-only scanner/trader
- Scans Top-100 by market cap (CMC) for signals (day/swing/trend)
- Scans Movers using legacy logic (15%+, volume, trend-confirm, fakeout-resistant)
- Excludes stable/pegged assets (symbol rules + price-band heuristic)
- Paper/live trading, max positions, Redis persistence
- Discord: Signals vs Trades channels, only post when there is content

ENV:
  REDIS_URL
  CMC_API_KEY
  DISCORD_SIGNALS_WEBHOOK
  DISCORD_TRADES_WEBHOOK
  MEXC_API_KEY, MEXC_SECRET (optional, for live trading)

Run:
  python3 mexc_trader_bot.py --config mexc_trader_bot_config.yml
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
        # smoke test
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")
    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])
    # positions / performance / portfolio
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

# ================== Stable/pegged filters ==================
STABLE_SYMBOLS = {
    "USDT","USDC","FDUSD","TUSD","DAI","USDD","USDP","PYUSD",
    "GUSD","USD1","USDE","EUR","EURS","XAUT","PAXG","XAU","XAU1",
}
PEGGED_KEYWORDS = [
    "USD","USDT","USDC","FDUSD","TUSD","USDD","USDP","PYUSD",
    "EUR","EURS","XAU","GOLD","PAXG","XAUT","USD1","USDE",
]
def is_symbol_stable_like(base_sym: str) -> bool:
    s = base_sym.upper()
    if s in STABLE_SYMBOLS:
        return True
    return any(k in s for k in PEGGED_KEYWORDS)
def is_price_pegged(df: pd.DataFrame, *, window: int = 72, band_pct: float = 3.0) -> bool:
    if df is None or len(df) < max(10, window//2):
        return False
    closes = df["close"].iloc[-min(window, len(df)):]
    mean_px = float(closes.mean())
    if mean_px <= 0:
        return False
    rng_pct = (float(closes.max()) - float(closes.min())) / mean_px * 100.0
    return rng_pct <= band_pct
def should_skip_stable_pair(client: "ExClient", pair: str) -> bool:
    base, _ = pair.split("/")
    if is_symbol_stable_like(base):
        return True
    try:
        df1h = client.ohlcv(pair, "1h", 96)
        return is_price_pegged(df1h, window=72, band_pct=3.0)
    except Exception:
        return False

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

# ================== BTC-independent Movers (legacy logic) ==================
def fetch_cmc_top_symbols(api_key: str, top_n: int = 100) -> List[str]:
    if not api_key: return []
    headers = {"X-CMC_PRO_API_KEY": api_key}
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": max(100, top_n), "convert": "USD", "sort": "market_cap", "sort_dir": "desc"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        data = r.json().get("data", [])
        syms = [it["symbol"].upper() for it in data[:top_n]]
        return syms
    except Exception:
        return []

def fetch_cmc_movers_legacy(cfg: Dict[str,Any]) -> List[str]:
    mv = cfg.get("movers", {})
    if not mv.get("enabled"): return []
    api_key = os.environ.get("CMC_API_KEY") or mv.get("cmc_api_key")
    if not api_key: 
        print("[movers] enabled but no CMC_API_KEY — skipping"); 
        return []
    headers = {"X-CMC_PRO_API_KEY": api_key}
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": mv.get("limit", 500), "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        data = r.json().get("data", [])
    except Exception as e:
        print("[movers] error:", e); return []

    min_change = mv.get("min_change_24h", 15.0)   # 24h %
    min_vol    = mv.get("min_volume_usd_24h", 5_000_000)
    max_age    = mv.get("max_age_days", 365)

    out=[]; now=pd.Timestamp.utcnow()
    for it in data:
        sym=it["symbol"].upper(); q=it.get("quote",{}).get("USD",{}); ch=(q.get("percent_change_24h") or 0); vol=(q.get("volume_24h") or 0)
        date_added=pd.to_datetime(it.get("date_added", now.isoformat()), utc=True); age_days=(now-date_added).days
        # legacy filters: strong up% + volume + not too old
        if (ch>=min_change) and (vol>=min_vol) and (age_days<=max_age):
            out.append(sym)
    return out

def filter_pairs_on_mexc(client: ExClient, symbols: List[str], quote: str="USDT") -> List[str]:
    client.load_markets()
    out=[]
    for sym in symbols:
        pair=f"{sym}/{quote}"
        if client.has_pair(pair): out.append(pair)
    return out

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

# ================== Paper performance snapshot ==================
def paper_snapshot(client: ExClient, rds: RedisState, pf: Dict[str,Any], perf: Dict[str,Any]):
    # live price for each holding
    mv_total = 0.0
    pnl_total = 0.0
    lines = []
    for sym, pos in sorted(pf.get("holdings", {}).items()):
        qty = float(pos["qty"]); avg = float(pos["avg"])
        px = client.last_price(sym)
        last = float(px) if px is not None else avg
        mv = qty * last
        pnl = (last - avg) * qty
        mv_total += mv
        pnl_total += pnl
        lines.append((sym, qty, avg, last, pnl, mv))
    equity = pf.get("cash_usdt", 0.0) + mv_total
    exposure = mv_total
    # Open R
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
    # closed stats — robust to missing 'outcome'
    closed = perf.get("closed_trades", [])
    win = 0.0; avgR = 0.0; medR = 0.0; bestR = 0.0; worstR = 0.0; pfactor = 0.0
    if closed:
        dfc = pd.DataFrame(closed)
        if "r_multiple" in dfc:
            win = float((dfc.get("r_multiple",0) > 0).mean() * 100.0)
            avgR = float(dfc.get("r_multiple",0).mean())
            medR = float(dfc.get("r_multiple",0).median())
            bestR = float(dfc.get("r_multiple",0).max())
            worstR = float(dfc.get("r_multiple",0).min())
            gains = dfc.loc[dfc.get("r_multiple",0)>0, "r_multiple"].sum()
            losses = -dfc.loc[dfc.get("r_multiple",0)<0, "r_multiple"].sum()
            pfactor = float(gains / losses) if losses > 0 else float("inf")
        elif "outcome" in dfc:
            win = float((dfc["outcome"].isin(["t1","t2"])).mean() * 100.0)
    # print snapshot
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

# ================== Discord helpers ==================
def post_discord(webhook: str, text: str):
    if not webhook or not text.strip():
        return
    try:
        requests.post(webhook, json={"content": text}, timeout=10)
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

    signals_wh = os.environ.get("DISCORD_SIGNALS_WEBHOOK","").strip()
    trades_wh  = os.environ.get("DISCORD_TRADES_WEBHOOK","").strip()

    # SD zones cache
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    zones_cache = {}

    # Params
    dayP   = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = TrendParams(**cfg.get("trend_trade_params", {}))

    # Load states
    positions = rds.load_positions()
    perf      = rds.load_perf()
    pf        = ensure_portfolio(rds, trading_cfg) if paper_mode else {}

    # ===== Build Top-100 pairs (exclude stables/pegs) =====
    syms100  = fetch_cmc_top_symbols(os.environ.get("CMC_API_KEY",""), 100)
    pairs100 = filter_pairs_on_mexc(client, syms100, "USDT")
    top_pairs = []
    for p in pairs100:
        base, _ = p.split("/")
        if is_symbol_stable_like(base): 
            continue
        if should_skip_stable_pair(client, p):
            continue
        top_pairs.append(p)

    # ===== Movers (legacy) -> map to MEXC and also exclude stables/pegs =====
    movers_syms = fetch_cmc_movers_legacy(cfg)
    movers_pairs_all = filter_pairs_on_mexc(client, movers_syms, "USDT")
    movers_pairs = []
    for p in movers_pairs_all:
        base, _ = p.split("/")
        if is_symbol_stable_like(base):
            continue
        if should_skip_stable_pair(client, p):
            continue
        movers_pairs.append(p)

    # zone cache (only for pairs we’ll scan)
    need_zones_pairs = list(dict.fromkeys(top_pairs + movers_pairs))
    if sd_cfg.get("enabled"):
        ztf = sd_cfg.get("timeframe_for_zones","1h"); zlook = sd_cfg.get("lookback",300)
        for pair in need_zones_pairs:
            try:
                zdf = client.ohlcv(pair, ztf, zlook)
                zones_cache[pair] = find_zones(
                    zdf, sd_cfg.get("impulse_factor",1.8), sd_cfg.get("zone_padding_pct",0.25), sd_cfg.get("max_age_bars",300)
                )
            except Exception as e:
                print(f"[zones] {pair} err:", e); zones_cache[pair]=[]

    # ===== Scan: Top-100 =====
    results_top = {"signals": []}

    # DAY (1h)
    for pair in top_pairs:
        try:
            if should_skip_stable_pair(client, pair): 
                continue
            df1h = client.ohlcv(pair, "1h", 300)
            zones = zones_cache.get(pair)
            sig = day_signal(df1h, dayP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc"})
                if should_skip_stable_pair(client, pair): 
                    continue
                results_top["signals"].append(sig)
        except Exception as e:
            print(f"[scan-top-day] {pair} err:", e)

    # SWING (4h)
    tf_swing = swingP.get("timeframe","4h")
    for pair in top_pairs:
        try:
            if should_skip_stable_pair(client, pair): 
                continue
            df4h = client.ohlcv(pair, tf_swing, 400)
            zones = zones_cache.get(pair)
            sig = swing_signal(df4h, swingP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":tf_swing,"exchange":"mexc"})
                if should_skip_stable_pair(client, pair): 
                    continue
                results_top["signals"].append(sig)
        except Exception as e:
            print(f"[scan-top-swing] {pair} err:", e)

    # TREND (1d)
    for pair in top_pairs:
        try:
            if should_skip_stable_pair(client, pair): 
                continue
            dfd = client.ohlcv(pair, "1d", 320)
            zones = zones_cache.get(pair)
            sig = trend_signal(dfd, trnP, sd_cfg, zones)
            if sig:
                sig.update({"symbol":pair,"timeframe":"1d","exchange":"mexc"})
                if should_skip_stable_pair(client, pair): 
                    continue
                results_top["signals"].append(sig)
        except Exception as e:
            print(f"[scan-top-trend] {pair} err:", e)

    # ===== Scan: Movers (legacy) =====
    results_movers = {"signals": []}
    for pair in movers_pairs:
        try:
            if should_skip_stable_pair(client, pair): 
                continue
            df1h = client.ohlcv(pair, "1h", 300)
            zones = zones_cache.get(pair)
            # Reuse day_signal as the final entry pattern, but mark the note and rely on legacy pre-filter we did
            sig = day_signal(df1h, dayP, sd_cfg, zones)
            if sig:
                sig["note"] = "Mover Trend"
                sig.update({"symbol":pair,"timeframe":"1h","exchange":"mexc"})
                if should_skip_stable_pair(client, pair): 
                    continue
                results_movers["signals"].append(sig)
        except Exception as e:
            print(f"[scan-movers] {pair} err:", e)

    # ===== EXECUTE entries (paper or live), but Signals channel shows all signals regardless of execution =====
    already_open_symbols = set(v["symbol"] for v in positions.get("active_positions", {}).values())
    n_open = len(already_open_symbols)
    signals_for_discord: List[Tuple[str, Dict[str,Any]]] = []  # ("Top100"/"Movers", sig)
    trade_msgs: List[str] = []

    def exec_signal(sig: Dict[str,Any]):
        nonlocal n_open
        sym = sig["symbol"]; tf = sig["timeframe"]; typ = sig["type"]; entry = sig["entry"]; stop = sig["stop"]
        # add to Signals listing (no “skipped/executed” annotation)
        signals_for_discord.append(("Top100", sig) if sig.get("note")!="Mover Trend" else ("Movers", sig))
        # execution constraints
        if sym in already_open_symbols: 
            return
        if n_open >= max_pos:
            return
        # execute
        if paper_mode:
            equity = pf.get("cash_usdt", 0.0)
            qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
            ok = paper_buy(rds, pf, sym, entry, qty, min_order)
            if ok:
                trade_msgs.append(f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `{entry:.6f}` (paper)")
            else:
                return
        else:
            try:
                px = entry * (1 + slip_bps/10000.0)
                bal = client.ex.fetch_balance().get("USDT", {}).get("free", 0.0)
                equity = float(bal)
                qty = compute_qty_for_risk(entry, stop, equity, risk_pct)
                notional = qty * px
                if notional < min_order or notional > equity:
                    return
                client.place_market(sym, "buy", qty)
                trade_msgs.append(f"• BUY `{sym}` {tf} qty `{qty:.6f}` @ `~{px:.6f}` (live)")
            except Exception as e:
                print(f"[live] order err {sym}:", e); 
                return

        # register state & perf
        signal_for_state = {"exchange":"mexc","symbol":sym,"type":typ if typ!="day" else "day","entry":entry,"timeframe":tf}
        update_positions_active(positions, signal_for_state)
        event_ts = sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat())
        add_open_trade(perf, exchange="mexc", symbol=sym, tf=tf, sig_type=typ, entry=entry, stop=stop,
                       t1=sig.get("t1"), t2=sig.get("t2"), event_ts=event_ts)
        already_open_symbols.add(sym)
        n_open += 1

    # Execute all signals (Top first, then movers)
    for s in results_top["signals"]: exec_signal(s)
    for s in results_movers["signals"]: exec_signal(s)

    # Save states
    rds.save_positions(positions)
    rds.save_perf(perf)

    # ===== Logs =====
    ts_now = pd.Timestamp.utcnow().isoformat()
    print(f"=== MEXC Signals @ {ts_now} ===")
    print(f"Scanned — 1h:{len(top_pairs)}  4h:{len(top_pairs)}  1d:{len(top_pairs)}  | movers:{len(movers_pairs)}")

    # Console pretty print
    def fmt_sig(s):
        return f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`"

    top_only = [s for tag,s in signals_for_discord if tag=="Top100"]
    mov_only = [s for tag,s in signals_for_discord if tag=="Movers"]

    if top_only:
        print("Signals — Top 100")
        for s in top_only: print(fmt_sig(s))
    if mov_only:
        print("Signals — Movers")
        for s in mov_only: print(fmt_sig(s))

    # ===== Discord: post signals (only if there are any) =====
    if signals_wh and signals_for_discord:
        parts=[]
        if top_only:
            parts.append("**Signals — Top 100**")
            for s in top_only:
                parts.append(f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`")
        if mov_only:
            parts.append("**Signals — Movers**")
            for s in mov_only:
                parts.append(f"• `{s['symbol']}` {s['timeframe']} *{s['type']}* — {s['note']}\n  entry `{s['entry']:.6f}` stop `{s['stop']:.6f}` t1 `{s.get('t1')}` t2 `{s.get('t2')}`")
        post_discord(signals_wh, "\n".join(parts))

    # ===== Discord: post trades (only if trades occurred) =====
    if trades_wh and trade_msgs:
        post_discord(trades_wh, "💼 **Trades**\n" + "\n".join(trade_msgs))

    # ===== Paper performance snapshot (each run) =====
    if paper_mode:
        paper_snapshot(client, rds, pf, perf)

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
