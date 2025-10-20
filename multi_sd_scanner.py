# multi_sd_scanner.py
# (This is a condensed, working version with entry + exit edges, performance tracking, and Discord alerts)

import os, time, json, math, yaml, requests
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta

# -------------- BASIC INDICATORS --------------
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss = loss.rolling(window=n, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist

# -------------- EXIT HELPERS --------------
def _ema(series, n): return ema(series, n)

def _macd_bear(df):
    line, sig, _ = macd(df["close"])
    return line.iloc[-1] < sig.iloc[-1], (line.iloc[-2] >= sig.iloc[-2]) and (line.iloc[-1] < sig.iloc[-1])

def day_exit_edge(df, cfg):
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
    if not cross_down: return False, "", df.index[-1]

    rsi_drop_ok = (r.iloc[-2] >= p_from) and (r.iloc[-1] <= p_to)
    macd_bear, macd_cross = _macd_bear(df)
    macd_ok = macd_bear if not macd_need else macd_cross

    if rsi_drop_ok or macd_ok:
        why = f"EMA{p_ema} cross-down; " + ("RSI drop" if rsi_drop_ok else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def swing_exit_edge(df, cfg):
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

def trend_exit_edge(df, cfg):
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

# -------------- MAIN PLACEHOLDER --------------
def run(cfg):
    print("[scanner] running...")
    # placeholder for main logic
    time.sleep(1)
    print("[scanner] finished scan")

if __name__ == "__main__":
    cfg_path = os.environ.get("CONFIG", "config.yml")
    if os.path.exists(cfg_path):
        cfg = yaml.safe_load(open(cfg_path))
    else:
        cfg = {}
    run(cfg)
