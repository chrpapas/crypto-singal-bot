#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, yaml, requests, time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np
import ccxt

print("=== multi_sd_scanner v2025-10-20 r2 ===", flush=True)

# --- helpers: parse comma-separated env/strings into list ---
def parse_csv_env(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    return []

def expand_env(obj):
    """Recursively expand ${VAR} in dicts/lists/strings using os.environ."""
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(x) for x in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], obj)
    return obj

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

# ================== Exit helper functions ==================
def _macd_bear(df):
    line, sig, _ = macd(df["close"])
    return line.iloc[-1] < sig.iloc[-1], (line.iloc[-2] >= sig.iloc[-2]) and (line.iloc[-1] < sig.iloc[-1])

def day_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 60: return False, "", None
    p_ema = int(cfg.get("ema_break", 20)); p_from = int(cfg.get("rsi_drop_from", 70)); p_to = int(cfg.get("rsi_drop_to", 60))
    macd_need = bool(cfg.get("macd_confirm", True))
    e = ema(df["close"], p_ema); r = rsi(df["close"], 14)
    close_now, close_prev = df["close"].iloc[-1], df["close"].iloc[-2]; e_now, e_prev = e.iloc[-1], e.iloc[-2]
    cross_down = (close_prev >= e_prev) and (close_now < e_now)
    if not cross_down: return False, "", df.index[-1]
    rsi_drop_ok = (r.iloc[-2] >= p_from) and (r.iloc[-1] <= p_to)
    macd_bear, macd_cross = _macd_bear(df); macd_ok = macd_bear if not macd_need else macd_cross
    if rsi_drop_ok or macd_ok:
        why = f"EMA{p_ema} cross-down; " + ("RSI drop" if rsi_drop_ok else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def swing_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 120: return False, "", None
    p_ema = int(cfg.get("ema_break", 50)); rsi_below = int(cfg.get("rsi_below", 50)); macd_need = bool(cfg.get("macd_confirm", True))
    e = ema(df["close"], p_ema); r = rsi(df["close"], 14)
    close_now, close_prev = df["close"].iloc[-1], df["close"].iloc[-2]; e_now, e_prev = e.iloc[-1], e.iloc[-2]
    cross_down = (close_prev >= e_prev) and (close_now < e_now)
    macd_bear, macd_cross = _macd_bear(df); macd_ok = macd_cross if macd_need else macd_bear
    if cross_down and (r.iloc[-1] <= rsi_below or macd_ok):
        why = f"EMA{p_ema} cross-down; " + ("RSI<=%d" % rsi_below if r.iloc[-1] <= rsi_below else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def trend_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 200: return False, "", None
    need_cross = bool(cfg.get("ema_cross_20_50", True)); rsi_below = int(cfg.get("rsi_below", 50)); macd_need = bool(cfg.get("macd_confirm", True))
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    cross20_50 = (e20.iloc[-2] >= e50.iloc[-2]) and (e20.iloc[-1] < e50.iloc[-1]) if need_cross else True
    r = rsi(df["close"], 14); macd_bear, macd_cross = _macd_bear(df); macd_ok = macd_cross if macd_need else macd_bear
    if cross20_50 and (r.iloc[-1] <= rsi_below or macd_ok):
        why = ("EMA20<EMA50 cross; " if need_cross else "") + ("RSI<=%d" % rsi_below if r.iloc[-1] <= rsi_below else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

# ================== main run() ==================
def run(cfg: Dict[str,Any]):
    # Expand ${VAR} placeholders from environment first
    cfg = expand_env(cfg)

    ex_names = parse_csv_env(cfg.get("exchanges") or os.environ.get("EXCHANGES","mexc"))
    watchlist = parse_csv_env(cfg.get("symbols_watchlist") or os.environ.get("SYMBOLS_WATCHLIST","BTC/USDT"))
    exits_cfg = cfg.get("exits", {"enabled": True})

    print("[scanner] exchanges:", ex_names, flush=True)
    print("[scanner] watchlist:", watchlist, flush=True)

    # Guard: if still placeholders, explain and bail
    if any(x.startswith("${") and x.endswith("}") for x in ex_names):
        print("[error] EXCHANGES not set. Set env EXCHANGES (e.g. 'mexc,gate,binance') or put explicit list in config.yml.", flush=True)
        return
    if any(x.startswith("${") and x.endswith("}") for x in watchlist):
        print("[error] SYMBOLS_WATCHLIST not set. Set env SYMBOLS_WATCHLIST (e.g. 'BTC/USDT,SOL/USDT') or put explicit list in config.yml.", flush=True)
        return

    ex_clients = {name: ExClient(name) for name in ex_names}

    for ex_name, client in ex_clients.items():
        for pair in watchlist:
            try:
                df1h = client.ohlcv(pair,"1h",200)
                ok, why, bar_ts = day_exit_edge(df1h, exits_cfg.get("day", {}))
                if ok: print(f"[EXIT] {ex_name} {pair} day-exit: {why}", flush=True)
                else: print(f"[OK] {ex_name} {pair} no exit", flush=True)
            except Exception as e:
                print(f"[exits] {ex_name} {pair} err:", e, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    if os.path.exists(args.config):
        with open(args.config,"r") as f: cfg = yaml.safe_load(f)
    else: cfg = {}
    run(cfg)
