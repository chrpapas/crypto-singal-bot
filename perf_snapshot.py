#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perf_snapshot.py — read scanner's performance from Redis and compute live stats
- Works with the same config.yml as your multi_sd_scanner_redis.py
- Fetches live prices via ccxt to mark open trades to market
- Prints: win rate, avg R, profit factor, expectancy, equity curve, max drawdown, etc.
"""

import os, json, argparse, yaml, math
from typing import Dict, Any
import pandas as pd
import numpy as np
import ccxt
import redis
from datetime import datetime

# ---------- helpers ----------
def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def expand(v):
        if isinstance(v, dict): return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list): return [expand(x) for x in v]
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.environ.get(v[2:-1], v)
        return v

    return expand(cfg)

def parse_list(v):
    if isinstance(v, list): return v
    if isinstance(v, str): return [s.strip() for s in v.split(",") if s.strip()]
    return []

def redis_key(prefix: str, *parts: str) -> str:
    return ":".join([prefix, *[str(p) for p in parts]])

def safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default

# ---------- core ----------
def compute_snapshot(cfg: Dict[str,Any]) -> None:
    # Redis
    p_cfg = cfg.get("persistence", {}) or {}
    redis_url = p_cfg.get("redis_url") or os.environ.get("REDIS_URL")
    prefix = p_cfg.get("key_prefix", "spideybot:v1")
    if not redis_url:
        raise RuntimeError("No Redis URL. Set persistence.redis_url or REDIS_URL")

    r = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=5)
    perf_json = r.get(redis_key(prefix, "state", "performance"))
    if not perf_json:
        print("No performance state found in Redis.")
        return

    perf = json.loads(perf_json)
    open_trades = perf.get("open_trades", [])
    closed_trades = perf.get("closed_trades", [])

    # Exchanges for marking to market
    ex_names = parse_list(cfg.get("exchanges") or "mexc")
    ex_map = {name: getattr(ccxt, name)({"enableRateLimit": True}) for name in ex_names}

    # mark-to-market for open trades
    mtm_rows = []
    for tr in open_trades:
        ex = ex_map.get(tr.get("exchange"))
        if ex is None:
            last = np.nan
        else:
            pair = tr.get("symbol")
            try:
                # fast path
                t = ex.fetch_ticker(pair)
                last = safe_float(t.get("last"))
                if math.isnan(last):
                    raise Exception("no last")
            except Exception:
                # fallback to timeframe close
                tf = tr.get("timeframe", "1h")
                try:
                    o = ex.fetch_ohlcv(pair, timeframe=tf, limit=1)
                    last = float(o[-1][4])
                except Exception:
                    last = np.nan

        entry = safe_float(tr.get("entry"))
        stop  = safe_float(tr.get("stop"))
        risk  = max(1e-12, entry - stop)
        r_now = (last - entry)/risk if (not math.isnan(last) and risk>0) else np.nan
        pct_now = (last/entry - 1.0)*100.0 if (not math.isnan(last) and entry>0) else np.nan
        mtm_rows.append({
            "id": tr.get("id"),
            "exchange": tr.get("exchange"),
            "symbol": tr.get("symbol"),
            "timeframe": tr.get("timeframe"),
            "type": tr.get("type"),
            "opened_at": tr.get("opened_at"),
            "entry": entry, "stop": stop, "last": last,
            "risk": risk, "R_unrealized": r_now, "pct_unrealized": pct_now
        })

    df_open = pd.DataFrame(mtm_rows) if mtm_rows else pd.DataFrame(columns=[
        "id","exchange","symbol","timeframe","type","opened_at",
        "entry","stop","last","risk","R_unrealized","pct_unrealized"
    ])
    df_closed = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame(columns=[
        "id","exchange","symbol","timeframe","type","opened_at","closed_at",
        "entry","stop","exit_price","outcome","r_multiple","pct_return","reason"
    ])

    # Clean types
    if not df_closed.empty:
        if "r_multiple" not in df_closed:
            df_closed["risk"] = (df_closed["entry"] - df_closed["stop"]).clip(lower=1e-12)
            df_closed["r_multiple"] = (df_closed["exit_price"] - df_closed["entry"]) / df_closed["risk"]
        df_closed["r_multiple"] = pd.to_numeric(df_closed["r_multiple"], errors="coerce")
        if "closed_at" in df_closed:
            df_closed["closed_at"] = pd.to_datetime(df_closed["closed_at"], errors="coerce", utc=True)

    # -------- metrics on closed --------
    def profit_factor(r):
        pos = r[r>0].sum()
        neg = -r[r<0].sum()
        return (pos/neg) if neg>0 else np.inf if pos>0 else 0.0

    if df_closed.empty:
        win_rate = avg_R = exp_R = pf = med_R = best_R = worst_R = 0.0
        trades_n = 0
    else:
        R = df_closed["r_multiple"].dropna()
        trades_n = len(R)
        win_rate = (R>0).mean()*100.0
        avg_R = R.mean()
        med_R = R.median()
        best_R = R.max()
        worst_R = R.min()
        pf = profit_factor(R)
        exp_R = avg_R

    # -------- equity curve + max drawdown (in R) --------
    def max_drawdown(series):
        if series.empty: return 0.0
        cummax = series.cummax()
        dd = series - cummax
        return float(dd.min())

    equity_R = pd.Series(dtype=float)
    if not df_closed.empty:
        dfc = df_closed.sort_values("closed_at")
        equity_R = dfc["r_multiple"].fillna(0).cumsum()

    unreal_R = float(df_open["R_unrealized"].dropna().sum()) if not df_open.empty else 0.0
    eq_with_open = equity_R.copy()
    if not eq_with_open.empty:
        eq_with_open.iloc[-1] = eq_with_open.iloc[-1] + unreal_R
    else:
        eq_with_open = pd.Series([unreal_R])

    mdd_R_closed_only = max_drawdown(equity_R) if not equity_R.empty else 0.0
    mdd_R_with_open   = max_drawdown(eq_with_open) if not eq_with_open.empty else 0.0

    # -------- breakdowns --------
    by_tf = None
    if not df_closed.empty:
        by = df_closed.groupby("timeframe")["r_multiple"]
        by_tf = by.agg(trades="count", win_rate=lambda s: (s>0).mean()*100.0,
                       avg_R="mean", pf=lambda s: profit_factor(s)).reset_index()

    # -------- print --------
    print("=== Performance Snapshot ===")
    print(f"As of: {datetime.utcnow().isoformat()}Z")
    print(f"Redis prefix: {prefix}")
    print("")
    print("--- Closed trades ---")
    print(f"Closed trades: {trades_n}")
    print(f"Win rate:      {win_rate:.1f}%")
    print(f"Avg R:         {avg_R:.3f}   | Median R: {med_R:.3f}")
    print(f"Best/Worst R:  {best_R:.2f} / {worst_R:.2f}")
    print(f"Profit factor: {pf:.2f}")
    print("")
    print("--- Open trades (MTM) ---")
    print(f"Open trades:   {len(df_open)}")
    print(f"Unrealized R:  {unreal_R:.3f}")
    print("")
    print("--- Equity (R) ---")
    latest_eq_closed = float(equity_R.iloc[-1]) if not equity_R.empty else 0.0
    latest_eq_with_open = float(eq_with_open.iloc[-1]) if not eq_with_open.empty else 0.0
    print(f"Equity (closed only): {latest_eq_closed:.2f} R | Max DD: {mdd_R_closed_only:.2f} R")
    print(f"Equity (+open MTM):   {latest_eq_with_open:.2f} R | Max DD: {mdd_R_with_open:.2f} R")
    print("")

    if by_tf is not None and not by_tf.empty:
        print("--- By timeframe ---")
        for _, row in by_tf.iterrows():
            print(f"{row['timeframe']:>3}: n={int(row['trades'])} | win {row['win_rate']:.1f}% | avgR {row['avg_R']:.3f} | PF {row['pf']:.2f}")
        print("")

    # optional: list open trades snapshot
    if not df_open.empty:
        print("--- Open trades detail (top 10 by |R|) ---")
        dfo = df_open.copy()
        dfo["absR"] = dfo["R_unrealized"].abs()
        view = dfo.sort_values("absR", ascending=False).head(10)[
            ["exchange", "symbol", "timeframe", "R_unrealized", "pct_unrealized", "entry", "last", "opened_at"]
        ]
        for _, r in view.iterrows():
            print(f"[{r['exchange']}] {r['symbol']} {r['timeframe']} | "
                  f"R={r['R_unrealized']:.2f} | {r['pct_unrealized']:.2f}% | "
                  f"entry={r['entry']:.6f} last={r['last']:.6f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    compute_snapshot(cfg)