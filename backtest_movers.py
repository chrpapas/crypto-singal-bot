# backtest_movers.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, math, json, time
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import ccxt

# ================= TA helpers =================
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff(); up = d.clip(lower=0).rolling(length).mean(); dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9); return 100 - (100 / (1 + rs))
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow); line = f - s; sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig; return line, sig, hist

# ================ Stable filters =================
DEFAULT_STABLES = {
    "USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT","USD1","USDE","USDY",
    "USDP","SUSD","EURS","EURT","PYUSD"
}
def is_stable_base(sym_pair: str, extra: List[str]) -> bool:
    base, _ = sym_pair.split("/")
    b = base.upper().replace("3L","").replace("5L","").replace("3S","").replace("5S","")
    extras = {e.upper() for e in (extra or [])}
    return (b in DEFAULT_STABLES) or (b in extras)

# ================ Mover signal =================
def mover_signal_1h(df: pd.DataFrame) -> Optional[Dict]:
    """
    Legacy mover breakout:
    - EMA20 > EMA50
    - MACD line > signal
    - RSI in [55,80]
    - Close > highest high of last 30 bars (excluding current)
    - Volume >= 20SMA
    Stop = min(last 10 lows)
    TP = +5%
    """
    if df is None or len(df) < 80: return None
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    mac_line, mac_sig, _ = macd(df["close"])
    r = rsi(df["close"],14); vS = sma(df["volume"], 20)
    last = df.iloc[-1]
    hl = df["high"].iloc[-31:-1].max()
    aligned = (e20.iloc[-1] > e50.iloc[-1]) and (mac_line.iloc[-1] > mac_sig.iloc[-1]) and (55 <= r.iloc[-1] <= 80)
    vol_ok = last["volume"] >= (vS.iloc[-1] if not np.isnan(vS.iloc[-1]) else last["volume"])
    breakout = last["close"] > hl
    if not (aligned and vol_ok and breakout): return None
    entry = float(last["close"]); stop = float(min(df["low"].iloc[-10:]))
    return {
        "type": "mover",
        "entry": entry,
        "stop": stop,
        "t1": round(entry*1.05, 12),
        "level": float(hl),
        "note": "Mover Trend",
        "event_bar_ts": df.index[-1].isoformat()
    }

# ================ CCXT helpers =================
def load_mexc_symbols(ex) -> List[str]:
    mkts = ex.load_markets()
    syms = []
    for k,m in mkts.items():
        if "/USDT" in k and m.get("spot", True):
            syms.append(k)
    return sorted(list(set(syms)))

def fetch_ohlcv_range(ex, symbol: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Pull all candles between start_ms and end_ms (inclusive) using CCXT pagination.
    """
    limit = 1000
    out = []
    since = start_ms
    while True:
        rows = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
        if not rows:
            break
        out.extend(rows)
        last_ts = rows[-1][0]
        # stop if we've reached the end
        if last_ts >= end_ms - 1:
            break
        # prevent infinite loop
        next_since = last_ts + 1
        if next_since <= since:
            break
        since = next_since
        # Rate limit guard
        time.sleep(ex.rateLimit/1000.0 if getattr(ex, "rateLimit", 0) else 0.2)
    if not out:
        return pd.DataFrame(columns=["open","high","low","close","volume"], index=pd.to_datetime([], utc=True))
    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    # CCXT sometimes returns duplicates; drop and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# ================ Rolling movers universe (per bar) =================
def rolling_quote_volume(df: pd.DataFrame, lookback: int) -> pd.Series:
    # approximate $ volume = close * volume; sum across last N bars
    return (df["close"] * df["volume"]).rolling(lookback).sum()

# ================ Backtest core =================
def backtest_movers(
    start: str,
    end: str,
    tf: str = "1h",
    min_usd_vol_lookback: int = 24,        # 24 bars ~ 1d on 1h TF
    min_usd_vol: float = 2_000_000.0,
    max_pairs: int = 120,
    extra_stables: List[str] = None,
    out_dir: str = "./backtest_out"
):
    extra_stables = extra_stables or []

    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc   = pd.Timestamp(end, tz="UTC")

    ex = ccxt.mexc({"enableRateLimit": True})
    symbols = load_mexc_symbols(ex)
    symbols = [s for s in symbols if not is_stable_base(s, extra_stables)]
    # slice to potential max_pairs (we’ll still filter by rolling volume later)
    symbols = symbols[:max_pairs]

    print(f"[universe] candidates -> {len(symbols)} /USDT spot pairs (ex-stables), pulling candles…")

    # Pull data for all symbols
    all_dfs: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_ohlcv_range(ex, sym, tf, int(start_utc.timestamp()*1000), int(end_utc.timestamp()*1000))
            if df.empty or len(df) < 100:
                continue
            all_dfs[sym] = df
        except Exception as e:
            # noisy exchanges / new listings, skip
            # print(f"[data] {sym} err: {e}")
            pass
        if i % 20 == 0:
            print(f"  pulled {i}/{len(symbols)}…")

    if not all_dfs:
        print("No data. Exiting.")
        return

    # Create a common timeline (union of all indices) then reindex each df forward-fill (NA safe for ta)
    common_idx = sorted(set().union(*[df.index for df in all_dfs.values()]))
    common_idx = pd.DatetimeIndex(common_idx, tz="UTC")
    # focus strictly on start..end boundaries (already fetched in range, but be safe)
    common_idx = common_idx[(common_idx >= start_utc) & (common_idx <= end_utc)]
    if len(common_idx) == 0:
        print("No bars in window after alignment.")
        return

    # Pre-compute rolling quote volume per symbol
    rqv: Dict[str, pd.Series] = {}
    for sym, df in all_dfs.items]:
        # reindex to common timeline, ffill ohlc as needed (volume NA -> 0)
        dfr = df.reindex(common_idx).copy()
        dfr[["open","high","low","close"]] = dfr[["open","high","low","close"]].ffill()
        dfr["volume"] = dfr["volume"].fillna(0)
        rqv[sym] = rolling_quote_volume(dfr, min_usd_vol_lookback)

    # Backtest state
    OPEN: Dict[str, Dict] = {}  # sym -> trade dict
    TRADES: List[Dict] = []
    equity = 0.0
    equity_curve = []

    print(f"[universe] {len(all_dfs)} pairs (movers-style) | tf={tf}")
    print(f"[movers] building timeline over {len(common_idx)} bars… (this may take a bit)")

    # Iterate over each bar (skip warmup bars)
    warmup = max(80, min_usd_vol_lookback + 10)
    for ts in common_idx[warmup:]:
        # Build movers universe for this bar: top by rolling $volume >= min_usd_vol
        current = []
        for sym, df in all_dfs.items():
            r = rqv[sym].reindex(common_idx).iloc[:common_idx.get_loc(ts)+1].iloc[-1]
            if r is not None and r >= min_usd_vol:
                current.append(sym)
        # Optional cap
        # current = current[:max_pairs]  # already capped earlier, but keep if needed

        # Generate/close per symbol
        for sym in current:
            dfr = all_dfs[sym].reindex(common_idx).loc[:ts].copy()
            # Ensure no NA at the edge
            dfr[["open","high","low","close"]] = dfr[["open","high","low","close"]].ffill()
            dfr["volume"] = dfr["volume"].fillna(0)

            # If trade is open, evaluate exit with this bar
            if sym in OPEN:
                tr = OPEN[sym]
                # Use bar’s high/low to check TP/SL touch
                hi = float(dfr.iloc[-1]["high"])
                lo = float(dfr.iloc[-1]["low"])
                hit = None; exit_px = None
                # Check stop first (conservative)
                if lo <= tr["stop"]:
                    hit = "stop"; exit_px = tr["stop"]
                # Check TP
                if hit is None and hi >= tr["t1"]:
                    hit = "t1"; exit_px = tr["t1"]
                if hit:
                    R = (exit_px - tr["entry"]) / max(1e-12, (tr["entry"] - tr["stop"]))
                    TRADES.append({
                        "symbol": sym, "opened_at": tr["opened_at"], "closed_at": ts.isoformat(),
                        "entry": tr["entry"], "stop": tr["stop"], "t1": tr["t1"],
                        "exit": exit_px, "outcome": hit, "r_multiple": R, "tf": tf, "type": "mover"
                    })
                    del OPEN[sym]
                    equity += R

            # If no open trade for this symbol, see if we have a fresh signal on this bar
            if sym not in OPEN:
                sig = mover_signal_1h(dfr)
                if sig:
                    entry = float(sig["entry"]); stop = float(sig["stop"]); t1 = float(sig["t1"])
                    if entry <= 0 or stop <= 0 or stop >= entry:
                        continue
                    OPEN[sym] = {
                        "entry": entry, "stop": stop, "t1": t1,
                        "opened_at": ts.isoformat()
                    }

        # mark equity
        equity_curve.append({"ts": ts.isoformat(), "equity_R": equity})

    # Close any still-open trades at the last close (mark as stop for conservative accounting)
    last_ts = common_idx[-1]
    for sym, tr in list(OPEN.items()):
        TRADES.append({
            "symbol": sym, "opened_at": tr["opened_at"], "closed_at": last_ts.isoformat(),
            "entry": tr["entry"], "stop": tr["stop"], "t1": tr["t1"],
            "exit": tr["stop"], "outcome": "stop", "r_multiple": (tr["stop"]-tr["entry"])/max(1e-12,(tr["entry"]-tr["stop"])),
            "tf": tf, "type": "mover"
        })
        del OPEN[sym]

    # ======= Stats =======
    trades_df = pd.DataFrame(TRADES)
    if trades_df.empty:
        print("No trades generated.")
        return
    wins = (trades_df["r_multiple"] > 0).sum()
    losses = (trades_df["r_multiple"] <= 0).sum()
    win_rate = (wins / max(1, len(trades_df))) * 100.0
    avgR = trades_df["r_multiple"].mean()
    bestR = trades_df["r_multiple"].max()
    worstR = trades_df["r_multiple"].min()

    gains = trades_df.loc[trades_df["r_multiple"] > 0, "r_multiple"].sum()
    losses_sum = -trades_df.loc[trades_df["r_multiple"] < 0, "r_multiple"].sum()
    pf = (gains / losses_sum) if losses_sum > 0 else float("inf")

    # Max drawdown on equity_R
    eq = pd.DataFrame(equity_curve)
    eq["equity_R"] = eq["equity_R"].astype(float)
    eq["peak"] = eq["equity_R"].cummax()
    dd = eq["equity_R"] - eq["peak"]
    max_dd = dd.min()

    os.makedirs(out_dir, exist_ok=True)
    trades_csv = os.path.join(out_dir, "movers_trades.csv")
    eq_csv = os.path.join(out_dir, "movers_equity_curve.csv")
    trades_df.to_csv(trades_csv, index=False)
    eq.to_csv(eq_csv, index=False)

    print("=== Backtest (Movers-only, legacy breakout, TP+5% / swing-low stop) ===")
    print(f"Trades: {len(trades_df)} | Win%: {win_rate:.1f}% | Avg R: {avgR:.2f} | PF: {pf if pf!=float('inf') else 'inf'} | Best/Worst R: {bestR:.2f}/{worstR:.2f}")
    print(f"Total R: {trades_df['r_multiple'].sum():.2f} | Max Drawdown: {max_dd:.2f} R")
    print(f"[saved] {trades_csv} and {eq_csv}")

# ================ CLI =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2025-07-01")
    ap.add_argument("--end", required=True, help="UTC end, e.g. 2025-10-29")
    ap.add_argument("--tf", default="1h", choices=["1h","4h"], help="timeframe (1h recommended for movers)")
    ap.add_argument("--min_usd_vol", type=float, default=2_000_000.0, help="rolling 24-bar $ volume threshold")
    ap.add_argument("--max_pairs", type=int, default=120, help="cap number of pairs to track")
    args = ap.parse_args()

    backtest_movers(
        start=args.start,
        end=args.end,
        tf=args.tf,
        min_usd_vol_lookback=24 if args.tf=="1h" else 6,  # ~24h window
        min_usd_vol=args.min_usd_vol,
        max_pairs=args.max_pairs,
        extra_stables=[],
        out_dir="./backtest_out"
    )

if __name__ == "__main__":
    main()
