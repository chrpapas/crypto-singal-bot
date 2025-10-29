#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_mexc_bot.py — signal backtester for the MEXC strategy

Features mirrored from the live bot:
- Bullish signals: day (1h), swing (4h), trend (1d)
- Bearish detectors (day/swing/trend) + "bear exit" rule for open bullish signals
- ETH regime gate for Top-100 longs (Movers longs are never gated)
- Universe: Top-100 by CMC (requires CMC_API_KEY) or custom list; optional fallback by MEXC top USDT-volume
- Outcome evaluation: T1-first, Stop-first, Bear-Exit, Timeout (max bars)
- R-multiples, per-bucket stats, CSV export of signals

Run example:
  python3 backtest_mexc_bot.py --config mexc_trader_bot_config.yml --universe top100 --tf 4h --start 2025-07-01 --end 2025-10-29 --gate on
"""

import argparse, os, json, math, time, sys
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import requests
import ccxt

# =============== TA helpers ===============
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl  = df['high'] - df['low']
    hc  = (df['high'] - df['close'].shift()).abs()
    lc  = (df['low']  - df['close'].shift()).abs()
    tr  = pd.concat([hl,hc,lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow)
    line = f - s
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# =============== Exchange wrapper (MEXC) ===============
class ExClient:
    def __init__(self):
        self.ex = ccxt.mexc({"enableRateLimit": True})
        key = os.environ.get("MEXC_API_KEY")
        sec = os.environ.get("MEXC_SECRET")
        if key and sec:
            self.ex.apiKey = key
            self.ex.secret = sec
        self._mkts = None
    def load_markets(self):
        if self._mkts is None:
            try: self._mkts = self.ex.load_markets()
            except Exception: self._mkts = {}
        return self._mkts
    def has_pair(self, symbol: str) -> bool:
        mkts = self.load_markets() or {}
        if symbol in mkts: return True
        syms = getattr(self.ex, "symbols", None) or []
        return symbol in syms
    def ohlcv(self, symbol: str, tf: str, since_ms: Optional[int]=None, limit: int=1000) -> pd.DataFrame:
        # fetch in batches to cover long ranges
        all_rows = []
        ms = since_ms
        max_iters = 20
        while True:
            try:
                rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, since=ms, limit=limit)
            except Exception:
                break
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < limit:
                break
            ms = rows[-1][0] + 1
            max_iters -= 1
            if max_iters <= 0:
                break
            time.sleep(self.ex.rateLimit/1000.0 if hasattr(self.ex, "rateLimit") else 0.2)
        if not all_rows:
            return pd.DataFrame(columns=["open","high","low","close","volume"], index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        return df

# =============== Stable filters (same as bot) ===============
DEFAULT_STABLES = {
    "USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT","USD1","USDE","USDY",
    "USDP","SUSD","EURS","EURT","PYUSD"
}
def is_stable_or_pegged(symbol: str, extra: List[str]) -> bool:
    base, _ = symbol.split("/")
    b = base.upper().replace("3L","").replace("5L","").replace("3S","").replace("5S","")
    extras = {e.upper() for e in (extra or [])}
    return (b in DEFAULT_STABLES) or (b in extras)

# =============== Signals (same logic as bot) ===============
def stop_from(df: pd.DataFrame, mode: str, atr_mult: float) -> float:
    if mode == "atr":
        a = atr(df, 14).iloc[-1]; a = 0 if np.isnan(a) else float(a)
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

def day_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None) -> Optional[Dict[str,Any]]:
    look, voln = int(p.get("lookback_high",30)), int(p.get("vol_sma",30))
    if len(df) < max(look, voln)+5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"]>volS.iloc[-1]) and (p.get("rsi_min",52)<=r.iloc[-1]<=p.get("rsi_max",78))
    retest_edge   = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok     = retest_edge and (last["volume"]>0.8*volS.iloc[-1]) and (r.iloc[-1]>=p.get("rsi_min",52))
    if not (breakout_ok or retrec_ok): return None
    entry = float(last["close"]); stop = stop_from(df, p.get("stop_mode","swing"), p.get("atr_mult",1.5))
    return {"type":"day","entry":entry,"stop":stop,"t1":entry*1.05,"t2":entry*1.10,"level":float(highlvl),
            "note":"Breakout" if breakout_ok else "Retest-Reclaim","event_idx": df.index[-1]}

def swing_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None) -> Optional[Dict[str,Any]]:
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
    entry=float(last['close']); stop=stop_from(df,p.get('stop_mode','swing'),p.get('atr_mult',2.0))
    return {"type":"swing","entry":entry,"stop":stop,"t1":entry*1.06,"t2":entry*1.12,"level":float(hl),
            "note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout", "event_idx": df.index[-1]}

def trend_signal(df: pd.DataFrame, p: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    need=max(p.get("ema100",100),p.get("vol_sma",20),p.get("breakout_lookback",55))+5
    if len(df)<need: return None
    df=df.copy(); df["ema20"]=ema(df["close"],p.get("ema20",20)); df["ema50"]=ema(df["close"],p.get("ema50",50))
    df["ema100"]=ema(df["close"],p.get("ema100",100)); df["volS"]=sma(df["volume"],p.get("vol_sma",20))
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1]>=p.get("rsi_min",50))
    within=abs((last["close"]-last["ema20"])/last["ema20"]*100)<=p.get("pullback_pct_max",10.0)
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(p.get("breakout_lookback",55)+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]>df["volS"].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    entry=float(last["close"]); stop=stop_from(df,p.get("stop_mode","swing"),p.get("atr_mult",2.0))
    return {"type":"trend","entry":entry,"stop":stop,"t1":entry*1.08,"t2":entry*1.20,"level":float(hl),
            "note":"Pullback-Bounce" if (within and bounce) else "Breakout","event_idx": df.index[-1]}

# Bearish detectors
def bear_day(df: pd.DataFrame, cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    tf = cfg.get("day", {})
    look = int(tf.get("lookback_low",20)); voln = int(tf.get("vol_sma",20))
    if len(df) < max(look, voln)+5: return None
    lowlvl = df["low"].iloc[-(look+1):-1].min(); last = df.iloc[-1]; volS = sma(df["volume"], voln); r = rsi(df["close"],14)
    breakdown = last["close"] < lowlvl if tf.get("require_breakdown", True) else (last["low"] < lowlvl)
    vol_ok = (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) if tf.get("require_vol_confirm", True) else True
    rsi_weak = r.iloc[-1] <= tf.get("rsi_max", 50)
    if not (breakdown and vol_ok and rsi_weak): return None
    return {"type":"bear_day","event_idx": df.index[-1]}
def bear_swing(df: pd.DataFrame, cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    tf = cfg.get("swing", {})
    if len(df) < 120: return None
    e20, e50, e100 = ema(df["close"],20), ema(df["close"],50), ema(df["close"],100)
    r = rsi(df["close"],14); last=df.iloc[-1]
    ema_bear = (e20.iloc[-1] < e50.iloc[-1] < e100.iloc[-1]) if tf.get("ema_stack_bear", True) else True
    lowlvl = df["low"].iloc[-(tf.get("lookback_low",34)+1):-1].min()
    breakdown = last["close"] < lowlvl; rsi_weak = r.iloc[-1] <= tf.get("rsi_max", 50)
    if not (ema_bear and breakdown and rsi_weak): return None
    return {"type":"bear_swing","event_idx": df.index[-1]}
def bear_trend(df: pd.DataFrame, cfg: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    tf = cfg.get("trend", {})
    if len(df) < 200: return None
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    r = rsi(df["close"],14); last=df.iloc[-1]
    cross = (e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-2] >= e50.iloc[-2]) if tf.get("ema20_below_50", True) else True
    lowlvl = df["low"].iloc[-(tf.get("lookback_low",55)+1):-1].min()
    breakdown = last["close"] < lowlvl; rsi_weak = r.iloc[-1] <= tf.get("rsi_max", 50)
    if not (cross and breakdown and rsi_weak): return None
    return {"type":"bear_trend","event_idx": df.index[-1]}

# =============== ETH regime gate ===============
def eth_gate_ok(client: ExClient, tf: str, start_ts_utc: pd.Timestamp, end_ts_utc: pd.Timestamp, gate_cfg: Dict[str,Any]) -> Dict[str,bool]:
    """Compute ON/OFF booleans for requested TF relative to ETH/USDT conditions."""
    out = {"1h": True, "4h": True, "1d": True}
    if not gate_cfg.get("enabled", True):
        return out
    try:
        df = client.ohlcv("ETH/USDT", tf, int(start_ts_utc.timestamp()*1000))
    except Exception:
        return out  # if we cannot fetch ETH, do not block; the live bot logs explicitly
    if df.empty:
        return out
    # evaluate only last bar inside range
    df = df.loc[start_ts_utc:end_ts_utc]
    if df.empty:
        return out
    e20 = ema(df["close"],20); e50 = ema(df["close"],50); rr = rsi(df["close"],14)
    last = -1
    ok = True
    if gate_cfg.get("require_ema_stack", True):
        ok &= (e20.iloc[last] > e50.iloc[last])
    if "min_rsi" in gate_cfg:
        ok &= (rr.iloc[last] >= gate_cfg.get("min_rsi", 50))
    if gate_cfg.get("require_breakout", False):
        lb = int(gate_cfg.get("lookback_breakout", 30))
        if len(df) >= lb+2:
            highlvl = df["high"].iloc[-(lb+1):-1].max()
            ok &= (df["close"].iloc[last] > highlvl)
    out[tf] = bool(ok)
    return out

# =============== Universe helpers ===============
def fetch_cmc_list(limit=140) -> List[dict]:
    api = os.environ.get("CMC_API_KEY","")
    if not api: 
        print("[cmc] no CMC_API_KEY; Top100 unavailable"); 
        return []
    try:
        r = requests.get(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
            headers={"X-CMC_PRO_API_KEY": api},
            params={"limit": limit, "convert": "USD"}, timeout=15
        )
        return r.json().get("data", [])
    except Exception as e:
        print("[cmc] fetch error:", e); 
        return []

def cmc_top_symbols(top_n=100) -> List[str]:
    data = fetch_cmc_list(limit=max(120, top_n+20))
    if not data: return []
    data.sort(key=lambda x: x.get("quote",{}).get("USD",{}).get("market_cap", 0), reverse=True)
    return [it.get("symbol","").upper() for it in data[:top_n]]

def mexc_filter_pairs(client: ExClient, symbols: List[str], quote="USDT", extra_stables: List[str]=None) -> List[str]:
    client.load_markets()
    out=[]
    for s in symbols:
        pair=f"{s}/{quote}"
        if client.has_pair(pair) and not is_stable_or_pegged(pair, extra_stables or []):
            out.append(pair)
    return out

def mexc_top_usdt_volume_pairs(client: ExClient, *, max_pairs=60, min_usd_vol=2_000_000, extra_stables=None) -> List[str]:
    extra_stables = extra_stables or []
    try:
        tickers = client.ex.fetch_tickers()
    except Exception as e:
        print("[fallback] fetch_tickers err:", e)
        return []
    items = []
    for sym, t in tickers.items():
        if "/USDT" not in sym: continue
        if is_stable_or_pegged(sym, extra_stables): continue
        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume"); last = t.get("last") or t.get("close")
            try: qv = (base_v or 0) * (last or 0)
            except Exception: qv = 0
        try: qv = float(qv or 0)
        except Exception: qv = 0.0
        if qv >= float(min_usd_vol):
            items.append((sym, qv))
    items.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s,_ in items[:max_pairs]]
    print(f"[fallback] MEXC top USDT-volume picked {len(pairs)} pairs (min_usd_vol={min_usd_vol})")
    return pairs

# =============== Evaluation helpers ===============
def evaluate_signal_forward(df: pd.DataFrame,
                            t0_idx: pd.Timestamp,
                            entry: float,
                            stop: float,
                            t1: Optional[float],
                            bear_trigger_indices: pd.DatetimeIndex,
                            max_forward_bars: int) -> Dict[str,Any]:
    """
    Walk forward bar-by-bar after t0 to find first event:
      - Stop hit (low <= stop)
      - T1 hit (high >= t1)     [if t1 is provided]
      - Bear exit if a bear signal time >= next bar and before stop/T1
      - Timeout (no event within max_forward_bars)
    """
    # start from bar strictly AFTER the signal bar
    if t0_idx not in df.index: 
        return {"outcome":"error","r_multiple":0.0}
    pos = df.index.get_loc(t0_idx)
    look = df.iloc[pos+1: pos+1+max_forward_bars]
    outcome=None; exit_px=None; bars_ahead=0
    for i, (ts, row) in enumerate(look.iterrows(), start=1):
        lo, hi, cl = float(row["low"]), float(row["high"]), float(row.get("close", np.nan))
        # check bear-exit first? In live we exit on bearish *appearance* before price checks for next decisions in same bar
        # To be conservative for evaluation, we prioritize stop/t1 intrabar; but if a bear signal bar occurs,
        # we mark bear_exit at the CLOSE of that bear bar (next available price).
        if t1 is not None and hi >= t1:
            outcome, exit_px, bars_ahead = "t1", t1, i
            break
        if lo <= stop:
            outcome, exit_px, bars_ahead = "stop", stop, i
            break
        # bear exit check (bear index equals this bar's ts)
        if ts in bear_trigger_indices:
            outcome, exit_px, bars_ahead = "bear_exit", cl if not math.isnan(cl) else entry, i
            break
    if outcome is None:
        outcome, exit_px, bars_ahead = "timeout", float(look["close"].iloc[-1]) if len(look) else entry, len(look)
    # R-multiple; use risk = entry - stop (>= tiny)
    risk = max(1e-9, entry - stop)
    r_mult = (exit_px - entry) / risk
    return {"outcome": outcome, "exit_px": exit_px, "bars": bars_ahead, "r_multiple": r_mult}

# =============== Main backtest ===============
def backtest(cfg: Dict[str,Any], universe: str, tf: str, start: str, end: str, gate_on: bool, pairs_arg: Optional[str], csv_out: Optional[str]):
    client = ExClient()
    extra_stables = cfg.get("filters", {}).get("extra_stables", [])
    dayP   = cfg.get("day_trade_params", {})
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trendP = cfg.get("trend_trade_params", {})

    bear_cfg = cfg.get("bearish_signals", {"enabled": True})
    eth_cfg  = cfg.get("eth_gate", {"enabled": True, "require_breakout": False, "require_ema_stack": True, "min_rsi": 50})

    start_utc = pd.to_datetime(start, utc=True)
    end_utc   = pd.to_datetime(end, utc=True)

    # Universe
    if pairs_arg:
        raw_pairs = [p.strip().upper() for p in pairs_arg.split(",") if p.strip()]
    elif universe == "top100":
        syms = cmc_top_symbols(100)
        if not syms:
            print("[universe] Top100 unavailable from CMC; supply --pairs to avoid fallback.")
        pairs = mexc_filter_pairs(client, syms, "USDT", extra_stables) if syms else []
        if not pairs:
            pairs = mexc_top_usdt_volume_pairs(client, max_pairs=60, min_usd_vol=2_000_000, extra_stables=extra_stables)
            print("[universe] Using MEXC volume fallback.")
        raw_pairs = pairs
    elif universe == "movers":
        # Backtester focuses on mechanics; for movers universe, re-use volume list unless user provides custom.
        raw_pairs = mexc_top_usdt_volume_pairs(client, max_pairs=60, min_usd_vol=2_000_000, extra_stables=extra_stables)
    else:
        raw_pairs = mexc_top_usdt_volume_pairs(client, max_pairs=60, min_usd_vol=2_000_000, extra_stables=extra_stables)

    # ETH gate value for this TF
    gate_tf_map = {"1h":"1h","4h":"4h","1d":"1d"}
    gate_tf = gate_tf_map.get(tf, "1h")
    gate_status = {"1h":True,"4h":True,"1d":True}
    if gate_on and eth_cfg.get("enabled", True):
        try:
            gate_status = eth_gate_ok(client, gate_tf, start_utc, end_utc, eth_cfg)
        except Exception:
            gate_status = {"1h":True,"4h":True,"1d":True}
    print(f"[eth-gate] {gate_tf}:{'ON' if gate_on and eth_cfg.get('enabled', True) else 'OFF'} -> status={gate_status.get(gate_tf, True)}")

    # Gather OHLCV & generate signals per pair
    all_records = []
    max_fwd = {"1h": 96, "4h": 60, "1d": 30}[tf]  # forward horizon for evaluation
    since_ms = int((start_utc - pd.Timedelta(days=120)).timestamp()*1000)  # pre-buffer for indicators

    for pair in raw_pairs:
        try:
            df = client.ohlcv(pair, tf, since_ms)
            if df.empty: 
                continue
            # trim to exact backtest window (tz-aware)
            df = df.loc[start_utc:end_utc]
            if len(df) < 100: 
                continue

            # Build bearish trigger index for bear-exit evaluation (on same tf)
            bear_idx = []
            if bear_cfg.get("enabled", True):
                for i in range(80, len(df)):
                    sub = df.iloc[:i+1]
                    b = None
                    if tf == "1h":
                        b = bear_day(sub, bear_cfg)
                    elif tf == "4h":
                        b = bear_swing(sub, bear_cfg)
                    elif tf == "1d":
                        b = bear_trend(sub, bear_cfg)
                    if b:
                        bear_idx.append(sub.index[-1])
            bear_index = pd.DatetimeIndex(bear_idx, tz="UTC")

            # Bullish signals (respect ETH gate for Top-100; Movers ignore gate)
            is_top = (universe == "top100" and pairs_arg is None)
            allow_longs = True
            if gate_on and eth_cfg.get("enabled", True) and is_top:
                allow_longs = bool(gate_status.get(gate_tf, True))

            for i in range(80, len(df)):
                sub = df.iloc[:i+1]
                ctx_time = sub.index[-1]
                # generate a bullish signal (if allowed)
                bull_sig=None
                if allow_longs:
                    if tf == "1h":
                        bull_sig = day_signal(sub, dayP, {}, None)
                    elif tf == "4h":
                        bull_sig = swing_signal(sub, swingP, {}, None)
                    elif tf == "1d":
                        bull_sig = trend_signal(sub, trendP)
                if bull_sig:
                    entry = float(bull_sig["entry"]); stop=float(bull_sig["stop"]); t1=float(bull_sig["t1"])
                    res = evaluate_signal_forward(df, ctx_time, entry, stop, t1, bear_index, max_fwd[tf])
                    all_records.append({
                        "symbol": pair, "tf": tf, "when": ctx_time.isoformat(),
                        "type": bull_sig["type"], "dir": "long",
                        "entry": entry, "stop": stop, "t1": t1,
                        **res
                    })

                # also mark bearish signals themselves for reporting (optional)
                if bear_cfg.get("enabled", True):
                    b=None
                    if tf == "1h": b = bear_day(sub, bear_cfg)
                    elif tf == "4h": b = bear_swing(sub, bear_cfg)
                    elif tf == "1d": b = bear_trend(sub, bear_cfg)
                    if b:
                        all_records.append({
                            "symbol": pair, "tf": tf, "when": ctx_time.isoformat(),
                            "type": b["type"], "dir": "short_signal",  # informational
                            "entry": np.nan, "stop": np.nan, "t1": np.nan,
                            "outcome": "bear_signal", "r_multiple": 0.0, "bars": 0
                        })

        except Exception as e:
            print(f"[pair] {pair} error:", e)
            continue

    if not all_records:
        print("No signals generated in the selected window.")
        return

    df_res = pd.DataFrame(all_records)
    # Summary
    longs = df_res[df_res["dir"]=="long"].copy()
    total = len(longs)
    t1n   = int((longs["outcome"]=="t1").sum())
    stopn = int((longs["outcome"]=="stop").sum())
    bexn  = int((longs["outcome"]=="bear_exit").sum())
    ton   = int((longs["outcome"]=="timeout").sum())
    avgR  = float(longs["r_multiple"].mean()) if total else 0.0
    medR  = float(longs["r_multiple"].median()) if total else 0.0
    winp  = (t1n/max(1,total))*100.0

    print("\n=== Backtest Summary ===")
    print(f"Universe size: {len(raw_pairs)} | TF: {tf} | Window: {start_utc.date()} -> {end_utc.date()}")
    print(f"Long signals: {total} | T1: {t1n} | Stop: {stopn} | Bear-Exit: {bexn} | Timeout: {ton}")
    print(f"Win% (T1-first): {winp:.1f}% | Avg R: {avgR:.2f} | Median R: {medR:.2f}")
    # By signal type
    if total:
        by_type = longs.groupby("type")["r_multiple"].agg(["count","mean","median"]).reset_index()
        print("\nBy signal type:")
        for _,row in by_type.iterrows():
            print(f"  {row['type']:>6}: n={int(row['count'])} | avgR={row['mean']:.2f} | medR={row['median']:.2f}")

    if csv_out:
        df_res.to_csv(csv_out, index=False)
        print(f"\nSaved detailed results -> {csv_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    ap.add_argument("--universe", default="top100", choices=["top100","movers","volume","custom"])
    ap.add_argument("--pairs", default="", help="Comma-separated list of pairs, e.g. BTC/USDT,ETH/USDT (overrides --universe)")
    ap.add_argument("--tf", default="4h", choices=["1h","4h","1d"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--gate", default="on", choices=["on","off"], help="ETH gate for Top-100 longs")
    ap.add_argument("--csv", default="", help="Optional CSV output path")
    args = ap.parse_args()

    # Load minimal config (dict-like); tolerate missing file
    cfg = {}
    if os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print("[config] load error:", e)
            cfg = {}

    gate_on = (args.gate.lower() == "on")
    backtest(cfg, args.universe, args.tf, args.start, args.end, gate_on, args.pairs if args.pairs else None, args.csv if args.csv else None)

if __name__ == "__main__":
    main()
