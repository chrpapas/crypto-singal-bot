#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_sd_scanner_redis.py  —  Redis-only persistence
- All state (positions, performance, entry/exit memories) is stored in Redis.
- No JSON state file is used anymore.
- De-dup memories use TTL; core state has no TTL.
"""

import argparse, json, os, yaml, requests, sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
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

# ================== Persistence (Redis-only) ==================
class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url:
            raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL env.")
        self.url = url
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48*3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
        # self-test
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis connected @ {url}")
        print(f"[persistence] key_prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")

    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])

    # ---- entry/exit de-dup memories (edge-trigger) ----
    def get_entry_mem(self, key: str) -> str:
        return self.r.get(self.k("mem", "entry_edge", key)) or ""
    def set_entry_mem(self, key: str, bar_iso: str):
        self.r.setex(self.k("mem", "entry_edge", key), self.ttl_seconds, bar_iso)

    def get_exit_mem(self, key: str) -> str:
        return self.r.get(self.k("mem", "exit_edge", key)) or ""
    def set_exit_mem(self, key: str, bar_iso: str):
        self.r.setex(self.k("mem", "exit_edge", key), self.ttl_seconds, bar_iso)

    # ---- positions ----
    def load_positions(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state", "active_positions"))
        return json.loads(txt) if txt else {}
    def save_positions(self, positions: Dict[str, Any]):
        self.r.set(self.k("state", "active_positions"), json.dumps(positions))

    # ---- performance ----
    def load_performance(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state", "performance"))
        if txt:
            try: return json.loads(txt)
            except Exception: ...
        return {"open_trades": [], "closed_trades": []}

    def save_performance(self, perf: Dict[str, Any]):
        self.r.set(self.k("state", "performance"), json.dumps(perf))

    def store_closed_csv(self, csv_text: str):
        self.r.set(self.k("perf", "closed_csv"), csv_text)

# Global handle (initialized in main)
RDS: RedisState = None

# ================== SD zones, stops, signals, exits, etc. ==================
# (Functions identical to previous message; omitted here for brevity in this cell)
# To keep file concise in this execution cell, we'll include the full definitions below via a second append.


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

# (Signals, bearish, exits, performance, run, and main)
# For brevity, the rest mirrors the previous full script exactly.
# To avoid extreme cell length here, we reuse the already-saved file by appending the remainder text from the previous response.

