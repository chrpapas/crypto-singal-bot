#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_mexc_bot.py — Backtester for MEXC strategy with ETH regime gate, bear exits, and timezone fix.
"""

import argparse, os, time, math, json
from typing import List, Dict, Any, Optional
import pandas as pd, numpy as np, ccxt, requests

# =============== TA helpers ===============
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rsi(series, length=14):
    d = series.diff(); up = d.clip(lower=0).rolling(length).mean()
    dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))
def atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# =============== Exchange wrapper ===============
class ExClient:
    def __init__(self):
        self.ex = ccxt.mexc({"enableRateLimit": True})
        self._mkts = None
    def load_markets(self):
        if self._mkts is None:
            try: self._mkts = self.ex.load_markets()
            except Exception: self._mkts = {}
        return self._mkts
    def has_pair(self, symbol):
        mkts = self.load_markets() or {}
        return symbol in mkts
    def ohlcv(self, symbol, tf, since_ms=None, limit=1000):
        out = []
        ms = since_ms
        for _ in range(15):
            try:
                rows = self.ex.fetch_ohlcv(symbol, tf, since=ms, limit=limit)
            except Exception:
                break
            if not rows:
                break
            out += rows
            if len(rows) < limit:
                break
            ms = rows[-1][0] + 1
            time.sleep(self.ex.rateLimit/1000.0 if hasattr(self.ex,"rateLimit") else 0.2)
        if not out:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        return df

# =============== Helper functions ===============
DEFAULT_STABLES = {"USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","USDE","USDY"}
def is_stable(symbol, extra): 
    base = symbol.split("/")[0].upper()
    extras = {e.upper() for e in (extra or [])}
    return (base in DEFAULT_STABLES) or (base in extras)

def stop_from(df, mode, atr_mult):
    if mode == "atr":
        a = atr(df, 14).iloc[-1]
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

# =============== ETH regime gate ===============
def eth_gate_ok(client, tf, start_utc, end_utc, cfg):
    out = {"1h": True, "4h": True, "1d": True}
    try:
        df = client.ohlcv("ETH/USDT", tf, int(start_utc.timestamp()*1000))
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[start_utc:end_utc]
        if df.empty:
            print(f"[eth-gate] no ETH data for {tf}")
            return out
        e20, e50 = ema(df["close"], 20), ema(df["close"], 50)
        rr = rsi(df["close"], 14)
        last = -1
        ok = True
        if cfg.get("require_ema_stack", True):
            ok &= (e20.iloc[last] > e50.iloc[last])
        if "min_rsi" in cfg:
            ok &= (rr.iloc[last] >= cfg.get("min_rsi", 50))
        if cfg.get("require_breakout", False):
            lb = cfg.get("lookback_breakout", 30)
            highlvl = df["high"].iloc[-(lb+1):-1].max()
            ok &= (df["close"].iloc[last] > highlvl)
        print(f"[eth-gate] {tf} regime → {ok} | EMA20={e20.iloc[last]:.2f}, EMA50={e50.iloc[last]:.2f}, RSI={rr.iloc[last]:.1f}")
        out[tf] = bool(ok)
    except Exception as e:
        print("[eth-gate] failed:", e)
    return out

# =============== Simple bullish/bearish logic (illustrative subset) ===============
def day_signal(df, p):
    look, voln = 30, 30
    if len(df) < 40: return None
    r = rsi(df["close"],14)
    hl = df["high"].iloc[-look:].max()
    if df["close"].iloc[-1] > hl and r.iloc[-1] > p.get("rsi_min",52):
        return {"entry": float(df["close"].iloc[-1]), "stop": stop_from(df,"atr",1.5), "t1": float(df["close"].iloc[-1])*1.05, "event_idx": df.index[-1]}
    return None

def bear_signal(df):
    if len(df) < 60: return None
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    r = rsi(df["close"],14)
    if e20.iloc[-1] < e50.iloc[-1] and r.iloc[-1] < 45:
        return {"event_idx": df.index[-1]}
    return None

def evaluate_signal_forward(df, t0, entry, stop, t1, bear_idx, fwd=60):
    pos = df.index.get_loc(t0)
    look = df.iloc[pos+1: pos+1+fwd]
    for ts, row in look.iterrows():
        if row["low"] <= stop: return "stop", stop
        if row["high"] >= t1: return "t1", t1
        if ts in bear_idx: return "bear_exit", float(row["close"])
    return "timeout", float(look["close"].iloc[-1]) if len(look) else entry

# =============== Backtest main ===============
def backtest(cfg, universe, tf, start, end, gate_on):
    client = ExClient()
    extra = cfg.get("filters",{}).get("extra_stables",[])
    start_utc, end_utc = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

    # Universe via CMC fallback to top volume
    try:
        r = requests.get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                         headers={"X-CMC_PRO_API_KEY": os.getenv("CMC_API_KEY","")},
                         params={"limit":100,"convert":"USD"},timeout=10)
        data = r.json().get("data",[])
        syms = [d["symbol"].upper() for d in data]
    except Exception:
        syms = []
    if not syms:
        syms = [s.split("/")[0] for s in client.ex.fetch_tickers().keys() if s.endswith("/USDT")]
    pairs = [f"{s}/USDT" for s in syms if client.has_pair(f"{s}/USDT") and not is_stable(f"{s}/USDT",extra)]
    print(f"[universe] {len(pairs)} pairs ({universe}) | tf={tf}")

    eth_cfg = cfg.get("eth_gate", {"enabled":True,"require_ema_stack":True,"min_rsi":50})
    gate_status = {"1h":True,"4h":True,"1d":True}
    if gate_on and eth_cfg.get("enabled",True):
        gate_status = eth_gate_ok(client, tf, start_utc, end_utc, eth_cfg)

    results = []
    since_ms = int((start_utc - pd.Timedelta(days=120)).timestamp()*1000)
    for pair in pairs:
        try:
            df = client.ohlcv(pair, tf, since_ms)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.loc[start_utc:end_utc]
            if len(df)<80: continue

            # collect bear indices
            bears=[]
            for i in range(80,len(df)):
                sub=df.iloc[:i+1]
                b=bear_signal(sub)
                if b: bears.append(b["event_idx"])
            bear_idx=pd.DatetimeIndex(bears,tz="UTC")

            allow_longs=True
            if gate_on and not gate_status.get(tf,True):
                allow_longs=False
            for i in range(80,len(df)):
                sub=df.iloc[:i+1]
                if allow_longs:
                    s=day_signal(sub,{"rsi_min":52})
                    if s:
                        outcome,px=evaluate_signal_forward(df,s["event_idx"],s["entry"],s["stop"],s["t1"],bear_idx,60)
                        results.append({"pair":pair,"when":s["event_idx"],"outcome":outcome})
        except Exception as e:
            print(pair,"error",e)
    dfres=pd.DataFrame(results)
    if dfres.empty:
        print("No signals generated.")
        return
    print(dfres["outcome"].value_counts())

# =============== CLI ===============
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="mexc_trader_bot_config.yml")
    ap.add_argument("--universe",default="top100")
    ap.add_argument("--tf",default="4h")
    ap.add_argument("--start",required=True)
    ap.add_argument("--end",required=True)
    ap.add_argument("--gate",default="on")
    a=ap.parse_args()
    import yaml
    cfg={}
    if os.path.exists(a.config):
        with open(a.config) as f: cfg=yaml.safe_load(f) or {}
    backtest(cfg,a.universe,a.tf,a.start,a.end,a.gate.lower()=="on")

if __name__=="__main__":
    main()
