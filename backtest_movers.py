#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_movers.py — backtest the *legacy_mover_signal* logic on MEXC spot

- Builds a high-liquidity /USDT universe from MEXC tickers (volume-based)
- Uses the same legacy_mover_signal (E20/E50 + MACD + RSI + breakout) as the live bot
- Entry = close of signal bar
- Exit = first hit of either stop or t1, with t1 taking priority when both hit in the same bar
- Outputs:
    ./backtest_out/movers_trades.csv
    ./backtest_out/movers_equity_curve.csv

Example:
  python3 backtest_movers.py --config mexc_trader_bot_config.yml --start 2025-09-01 --end 2025-10-29 --tf 1h
"""

import argparse, os
from typing import Dict, Any, List, Optional

import ccxt
import numpy as np
import pandas as pd
import yaml

# ===== TA helpers (must match live bot) =====
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast)
    s = ema(series, slow)
    line = f - s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# ===== Stable filtering (same as bot) =====
DEFAULT_STABLES = {
    "USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT","USD1","USDE","USDY",
    "USDP","SUSD","EURS","EURT","PYUSD"
}

def is_stable_or_pegged(symbol: str, extra: List[str]) -> bool:
    base, _ = symbol.split("/")
    b = base.upper().replace("3L", "").replace("3S", "").replace("5L", "").replace("5S", "")
    extras = {e.upper() for e in (extra or [])}
    return (b in DEFAULT_STABLES) or (b in extras)

# ===== Legacy movers signal (copied from bot) =====
def legacy_mover_signal(df1h: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Exact same logic as in mexc_trader_bot.py

    Conditions:
    - E20 > E50
    - MACD line > signal
    - 55 <= RSI(14) <= 80
    - Close > 31-bar high (excluding current bar)
    - Volume >= SMA20(volume)
    """
    if df1h is None or len(df1h) < 80:
        return None

    e20, e50 = ema(df1h["close"], 20), ema(df1h["close"], 50)
    mac_line, mac_sig, _ = macd(df1h["close"])
    r = rsi(df1h["close"], 14)
    vS = sma(df1h["volume"], 20)

    last = df1h.iloc[-1]
    hl = df1h["high"].iloc[-31:-1].max()

    aligned = (
        (e20.iloc[-1] > e50.iloc[-1])
        and (mac_line.iloc[-1] > mac_sig.iloc[-1])
        and (55 <= r.iloc[-1] <= 80)
    )

    breakout = (
        last["close"] > hl
        and last["volume"] >= (vS.iloc[-1] if not np.isnan(vS.iloc[-1]) else last["volume"])
    )

    if not (aligned and breakout):
        return None

    entry = float(last["close"])
    stop = float(min(df1h["low"].iloc[-10:]))
    t1 = round(entry * 1.05, 6)
    t2 = round(entry * 1.10, 6)

    return {
        "type": "day",
        "entry": entry,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "level": float(hl),
        "note": "Mover Trend",
        "event_bar_ts": df1h.index[-1].isoformat(),
    }

# ===== Universe & data =====
def build_universe(
    ex: ccxt.Exchange,
    extra_stables: List[str],
    min_usd_vol: float = 2_000_000,
    max_pairs: int = 120,
) -> List[str]:
    print("[universe] candidates -> scanning /USDT spot tickers…")
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:
        print("[error] fetch_tickers:", e)
        return []

    items = []
    for sym, t in tickers.items():
        if "/USDT" not in sym:
            continue
        if is_stable_or_pegged(sym, extra_stables):
            continue
        # skip non-spot if ccxt marks them
        if t.get("spot") is False:
            continue

        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume")
            last = t.get("last") or t.get("close")
            try:
                qv = (base_v or 0) * (last or 0)
            except Exception:
                qv = 0

        try:
            qv = float(qv or 0)
        except Exception:
            qv = 0.0

        if qv >= min_usd_vol:
            items.append((sym, qv))

    items.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s, _ in items[:max_pairs]]
    print(f"[universe] volume universe -> {len(pairs)} /USDT spot pairs (ex-stables), min_vol=${min_usd_vol:,.0f}")
    return pairs

def fetch_all_ohlcv(
    ex: ccxt.Exchange,
    pairs: List[str],
    tf: str,
    start: str,
    end: str,
    warmup_bars: int = 200,
) -> Dict[str, pd.DataFrame]:
    start_ts = pd.to_datetime(start).tz_localize("UTC")
    end_ts = pd.to_datetime(end).tz_localize("UTC")

    tf_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(tf, 60)

    total_minutes = (end_ts - start_ts).total_seconds() / 60.0
    need_bars = int(total_minutes / tf_minutes) + warmup_bars + 10
    limit = max(need_bars, warmup_bars + 50)

    all_dfs: Dict[str, pd.DataFrame] = {}
    n = len(pairs)
    print(f"[data] fetching OHLCV for {n} pairs, tf={tf}, limit~{limit} bars each…")

    for idx, sym in enumerate(pairs, start=1):
        try:
            rows = ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts")
            df = df.loc[start_ts:end_ts]
            if len(df) < warmup_bars:
                continue
            all_dfs[sym] = df
        except Exception as e:
            print(f"[warn] ohlcv error {sym}: {e}")

        if idx % 20 == 0 or idx == n:
            print(f"  pulled {idx}/{n}…")

    print(f"[universe] {len(all_dfs)} pairs with sufficient history | tf={tf}")
    return all_dfs

# ===== Trade simulation =====
def simulate_symbol(sym: str, df: pd.DataFrame, tf: str) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    if len(df) < 120:
        return trades

    trade_open: Optional[Dict[str, Any]] = None
    entry_idx: Optional[int] = None

    for i, (ts, row) in enumerate(df.iterrows()):
        if i < 80:
            continue

        # 1) manage open trade using same priority logic as close_silent_if_hit
        if trade_open is not None and entry_idx is not None and i > entry_idx:
            lo = float(row["low"])
            hi = float(row["high"])
            stop = trade_open["stop"]
            t1 = trade_open["t1"]
            outcome = None
            exit_price = None

            # if both stop and t1 in same bar, count as t1 (favourable)
            if (lo <= stop) and (t1 is not None and hi >= t1):
                if hi >= t1:
                    outcome, exit_price = "t1", t1
                else:
                    outcome, exit_price = "stop", stop
            elif lo <= stop:
                outcome, exit_price = "stop", stop
            elif t1 is not None and hi >= t1:
                outcome, exit_price = "t1", t1

            if outcome:
                risk = trade_open["entry"] - trade_open["stop"]
                r_mult = (exit_price - trade_open["entry"]) / risk if risk != 0 else 0.0
                trades.append(
                    {
                        "symbol": sym,
                        "timeframe": trade_open["timeframe"],
                        "type": trade_open["type"],
                        "open_time": trade_open["open_time"],
                        "close_time": ts,
                        "entry": trade_open["entry"],
                        "stop": trade_open["stop"],
                        "t1": trade_open["t1"],
                        "exit_price": exit_price,
                        "outcome": outcome,
                        "r_multiple": r_mult,
                        "hold_bars": i - entry_idx,
                    }
                )
                trade_open = None
                entry_idx = None
                # don't open a new trade on the same bar as exit
                continue

        # 2) no open trade -> check for new signal using legacy_mover_signal on history up to this bar
        if trade_open is None:
            window = df.iloc[: i + 1]
            sig = legacy_mover_signal(window)
            if sig is not None:
                trade_open = {
                    "timeframe": tf,
                    "type": sig["type"],
                    "open_time": ts,
                    "entry": sig["entry"],
                    "stop": sig["stop"],
                    "t1": sig["t1"],
                }
                entry_idx = i

    return trades

# ===== Backtest runner =====
def backtest(start: str, end: str, tf: str, cfg_path: Optional[str] = None):
    cfg: Dict[str, Any] = {}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    extra_stables = cfg.get("filters", {}).get("extra_stables", [])

    ex = ccxt.mexc({"enableRateLimit": True})

    pairs = build_universe(
        ex,
        extra_stables=extra_stables,
        min_usd_vol=2_000_000,
        max_pairs=120,
    )

    all_dfs = fetch_all_ohlcv(
        ex,
        pairs,
        tf=tf,
        start=start,
        end=end,
        warmup_bars=120,
    )

    print("[movers] running legacy_mover_signal backtest…")
    all_trades: List[Dict[str, Any]] = []
    for sym, df in all_dfs.items():
        sym_trades = simulate_symbol(sym, df, tf=tf)
        all_trades.extend(sym_trades)

    if not all_trades:
        print("No trades generated in this window.")
        return

    df_tr = pd.DataFrame(all_trades)
    n = len(df_tr)
    wins = (df_tr["r_multiple"] > 0).sum()
    losses = (df_tr["r_multiple"] < 0).sum()
    win_rate = wins / n * 100.0
    avgR = df_tr["r_multiple"].mean()
    bestR = df_tr["r_multiple"].max()
    worstR = df_tr["r_multiple"].min()

    gains = df_tr.loc[df_tr["r_multiple"] > 0, "r_multiple"].sum()
    losses_sum = -df_tr.loc[df_tr["r_multiple"] < 0, "r_multiple"].sum()
    pf = gains / losses_sum if losses_sum > 0 else float("inf")

    eq = df_tr["r_multiple"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()

    print("=== Backtest (Movers-only, legacy_mover_signal, TP+5% / swing-low stop) ===")
    print(
        f"Trades: {n} | Win%: {win_rate:.1f}% | Avg R: {avgR:.2f} | PF: {pf:.2f} | "
        f"Best/Worst R: {bestR:.2f}/{worstR:.2f}"
    )
    print(f"Total R: {df_tr['r_multiple'].sum():.2f} | Max Drawdown: {max_dd:.2f} R")

    os.makedirs("./backtest_out", exist_ok=True)
    out_trades = "./backtest_out/movers_trades.csv"
    out_eq = "./backtest_out/movers_equity_curve.csv"
    df_tr.to_csv(out_trades, index=False)
    eq_df = pd.DataFrame({"trade_idx": range(1, n + 1), "equity_R": eq})
    eq_df.to_csv(out_eq, index=False)
    print(f"[saved] {out_trades} and {out_eq}")

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to mexc_trader_bot_config.yml (optional)")
    ap.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    ap.add_argument("--tf", default="1h", help="Timeframe (default 1h)")
    args = ap.parse_args()

    backtest(start=args.start, end=args.end, tf=args.tf, cfg_path=args.config)

if __name__ == "__main__":
    main()
