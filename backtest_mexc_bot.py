#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_mexc_bot.py — Backtester for the MEXC strategy with:
- ETH regime gate (rolling per-bar evaluation)
- Bullish day-signal logic (breakout) + bear exits
- Timezone-safe slicing (UTC everywhere)
- CMC Top100 universe with MEXC fallback by USDT volume
- Detailed per-trade logs to CSV + summary metrics

Example:
  python3 backtest_mexc_bot.py \
    --config mexc_trader_bot_config.yml \
    --universe top100 \
    --tf 4h \
    --start 2025-07-01 \
    --end 2025-10-29 \
    --gate on \
    --out trades.csv
"""

import argparse, os, time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd, numpy as np, ccxt, requests, yaml

# ================= TA helpers =================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rsi(series, length=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))
def atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ================= Exchange wrapper =================
class ExClient:
    def __init__(self):
        self.ex = ccxt.mexc({"enableRateLimit": True})
        self._mkts = None
        self.rate = getattr(self.ex, "rateLimit", 200)
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
        for _ in range(18):
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
            time.sleep(self.rate/1000.0)
        if not out:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        return df

# ================= Helpers =================
DEFAULT_STABLES = {"USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","USDE","USDY","PYUSD","USDP"}
def is_stable(symbol, extra):
    base = symbol.split("/")[0].upper()
    extras = {e.upper() for e in (extra or [])}
    return (base in DEFAULT_STABLES) or (base in extras)

def stop_from(df, mode="atr", atr_mult=1.5):
    if mode == "atr":
        a = atr(df, 14).iloc[-1]
        a = 0.0 if pd.isna(a) else float(a)
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

# ================= ETH regime gate (rolling) =================
def build_eth_gate_series(client: "ExClient", tf: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, cfg: Dict[str,Any]) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by bar time with columns:
      gate_ok: bool
      ema20, ema50, rsi
    Evaluated per bar so entries are tested at their specific timestamp.
    """
    try:
        df = client.ohlcv("ETH/USDT", tf, int((start_utc - pd.Timedelta(days=60)).timestamp()*1000))
        if df.empty:
            print(f"[eth-gate] no ETH data for {tf}; gate will be OFF.")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[:end_utc]  # include history before start to compute indicators
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["rsi"]   = rsi(df["close"], 14)

        require_ema = cfg.get("require_ema_stack", True)
        min_rsi     = cfg.get("min_rsi", 50)
        if cfg.get("require_breakout", False):
            lb = int(cfg.get("lookback_breakout", 30))
            highlvl = df["high"].rolling(lb, min_periods=lb).max().shift(1)
            cond_break = df["close"] > highlvl
        else:
            cond_break = pd.Series(True, index=df.index)

        cond_ema = (df["ema20"] > df["ema50"]) if require_ema else pd.Series(True, index=df.index)
        cond_rsi = (df["rsi"] >= min_rsi) if ("min_rsi" in cfg) else pd.Series(True, index=df.index)

        df["gate_ok"] = (cond_ema & cond_rsi & cond_break).fillna(False)
        # Only keep the backtest window
        return df.loc[start_utc:end_utc][["gate_ok","ema20","ema50","rsi"]]
    except Exception as e:
        print("[eth-gate] failed:", e)
        return pd.DataFrame()

# ================= Signal logic (simplified) =================
def day_long_signal(df, rsi_min=52, lookback_high=30, atr_mult=1.5):
    if len(df) < max(lookback_high, 40): return None
    last, prev = df.iloc[-1], df.iloc[-2]
    r = rsi(df["close"], 14)
    volS = sma(df["volume"], 30)
    highlvl = df["high"].iloc[-(lookback_high+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"] > (volS.iloc[-1] if not pd.isna(volS.iloc[-1]) else last["volume"])) and (r.iloc[-1] >= rsi_min)
    if not breakout_ok: return None
    entry = float(last["close"])
    stop  = float(stop_from(df, "atr", atr_mult))
    t1    = float(entry * 1.05)
    return {"entry":entry,"stop":stop,"t1":t1,"open_ts": df.index[-1]}

def bear_trigger(df) -> bool:
    """Simple bear trigger for exits: EMA20 < EMA50 and RSI < 45."""
    if len(df) < 60: return False
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    r = rsi(df["close"],14)
    return bool((e20.iloc[-1] < e50.iloc[-1]) and (r.iloc[-1] < 45))

# ================= Universe builders =================
def cmc_top_symbols(top_n=100) -> List[str]:
    key = os.getenv("CMC_API_KEY","")
    if not key: return []
    try:
        r = requests.get(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
            headers={"X-CMC_PRO_API_KEY": key},
            params={"limit": max(100, top_n), "convert": "USD", "sort": "market_cap", "sort_dir": "desc"},
            timeout=12
        )
        data = r.json().get("data", [])
        return [it["symbol"].upper() for it in data[:top_n] if "symbol" in it]
    except Exception:
        return []

def mexc_top_usdt_volume_pairs(client: ExClient, *, max_pairs=60, min_usd_vol=2_000_000, extra_stables=None):
    extra_stables = extra_stables or []
    try:
        tickers = client.ex.fetch_tickers()
    except Exception as e:
        print("[fallback] fetch_tickers err:", e)
        return []
    items = []
    for sym, t in tickers.items():
        if "/USDT" not in sym: continue
        if is_stable(sym, extra_stables): continue
        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume")
            last = t.get("last") or t.get("close")
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

# ================= Evaluation =================
def evaluate_trade_forward(df: pd.DataFrame, start_idx: pd.Timestamp, entry: float, stop: float, t1: float,
                           max_bars: int, fee_bps: float, slip_bps: float) -> Tuple[str, float, pd.Timestamp, int]:
    """
    Returns (outcome, exit_price_after_costs, exit_ts, hold_bars)
    """
    try:
        pos = df.index.get_loc(start_idx)
    except KeyError:
        pos = df.index.searchsorted(start_idx)
    look = df.iloc[pos+1 : pos+1+max_bars]
    if look.empty:
        return ("timeout", entry, df.index[min(pos, len(df)-1)], 0)

    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0
    buy_price  = entry * (1 + slip + fee)  # kept for clarity

    for i, (ts, row) in enumerate(look.iterrows(), start=1):
        if row["low"] <= stop:
            exit_price = stop * (1 - slip - fee)
            return ("stop", exit_price, ts, i)
        if row["high"] >= t1:
            exit_price = t1 * (1 - slip - fee)
            return ("t1", exit_price, ts, i)
        sub = df.loc[:ts]
        if bear_trigger(sub):
            exit_price = float(row["close"]) * (1 - slip - fee)
            return ("bear_exit", exit_price, ts, i)

    last_ts = look.index[-1]
    exit_price = float(look["close"].iloc[-1]) * (1 - slip - fee)
    return ("timeout", exit_price, last_ts, len(look))

def r_multiple(entry: float, stop: float, exit_price: float) -> float:
    risk = max(1e-9, entry - stop)
    return (exit_price - entry) / risk

# ================= Backtest core =================
def backtest(cfg: Dict[str,Any], universe: str, tf: str, start: str, end: str,
             gate_on: bool, out_csv: str, fee_bps: float, slip_bps: float,
             max_hold_bars: int):

    client = ExClient()
    extra = cfg.get("filters",{}).get("extra_stables",[])
    start_utc, end_utc = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)

    # Universe
    pairs: List[str] = []
    if universe.lower() == "top100":
        syms = cmc_top_symbols(100)
        if not syms:
            print("[universe] CMC failed; using MEXC volume fallback.")
            pairs = mexc_top_usdt_volume_pairs(client, max_pairs=60, min_usd_vol=2_000_000, extra_stables=extra)
        else:
            pairs = [f"{s}/USDT" for s in syms if client.has_pair(f"{s}/USDT") and not is_stable(f"{s}/USDT", extra)]
    else:
        pairs = mexc_top_usdt_volume_pairs(client, max_pairs=60, min_usd_vol=2_000_000, extra_stables=extra)

    print(f"[universe] {len(pairs)} pairs ({universe}) | tf={tf}")

    # Build rolling ETH gate series once
    eth_cfg = cfg.get("eth_gate", {"enabled": True, "require_ema_stack": True, "min_rsi": 50})
    eth_gate_df = pd.DataFrame()
    
    eth_enabled = bool(eth_cfg.get("enabled", True))
    
    if eth_enabled and gate_on:
        eth_gate_df = build_eth_gate_series(client, tf, start_utc, end_utc, eth_cfg)
        if eth_gate_df.empty:
            print("[eth-gate] series empty; gate will effectively block longs.")
        else:
            last = eth_gate_df.iloc[-1]
            print(f"[eth-gate] sample -> gate_ok(last)={bool(last.get('gate_ok', False))} | EMA20={last['ema20']:.2f} EMA50={last['ema50']:.2f} RSI={last['rsi']:.1f}")
    else:
        print("[eth-gate] OFF (config disabled or CLI off)")

    # Backtest
    trades: List[Dict[str,Any]] = []

    # pull a bit more history to compute indicators
    backpad_days = {"5m": 5, "15m": 10, "30m": 20, "45m": 25, "1h": 40, "4h": 90, "1d": 400}.get(tf, 60)
    since_ms = int((start_utc - pd.Timedelta(days=backpad_days)).timestamp() * 1000)

    for pair in pairs:
        try:
            df = client.ohlcv(pair, tf, since_ms)
            if df.empty:
                continue
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.loc[start_utc:end_utc]
            if len(df) < 120:
                continue

            for i in range(60, len(df)-1):
                sub = df.iloc[:i+1]
                ts_open = sub.index[-1]

                # Rolling ETH gate check at ts_open
                if gate_on and eth_cfg.get("enabled", True):
                    if eth_gate_df.empty:
                        continue
                    j = eth_gate_df.index.searchsorted(ts_open)
                    if j == len(eth_gate_df):
                        gate_ok = bool(eth_gate_df["gate_ok"].iloc[-1])
                    else:
                        # if exact time not present, use previous bar gate
                        gate_ok = bool(eth_gate_df["gate_ok"].iloc[max(0, j-1)])
                    if not gate_ok:
                        continue

                sig = day_long_signal(
                    sub,
                    rsi_min=cfg.get("day_trade_params",{}).get("rsi_min", 52),
                    lookback_high=cfg.get("day_trade_params",{}).get("lookback_high", 30),
                    atr_mult=cfg.get("day_trade_params",{}).get("atr_mult", 1.5)
                )
                if not sig:
                    continue

                entry = float(sig["entry"])
                stop  = float(sig["stop"])
                t1    = float(sig["t1"])

                outcome, exit_px, ts_exit, hold_bars = evaluate_trade_forward(
                    df, ts_open, entry, stop, t1, max_bars=max_hold_bars,
                    fee_bps=fee_bps, slip_bps=slip_bps
                )

                rr = r_multiple(entry, stop, exit_px)
                pnl_pct = (exit_px / entry - 1.0) * 100.0

                trades.append({
                    "pair": pair, "tf": tf,
                    "open_ts": ts_open.isoformat(),
                    "close_ts": ts_exit.isoformat(),
                    "hold_bars": hold_bars,
                    "entry": entry, "stop": stop, "t1": t1,
                    "exit": float(exit_px),
                    "outcome": outcome,
                    "r_multiple": float(rr),
                    "pnl_pct": float(pnl_pct),
                })

        except Exception as e:
            print(f"[{pair}] error:", e)

    if not trades:
        print("No signals generated in the selected window.")
        return

    dftr = pd.DataFrame(trades)
    dftr.sort_values("open_ts", inplace=True)
    dftr.to_csv(out_csv, index=False)
    print(f"[output] wrote {len(dftr)} trades -> {out_csv}")

    # Summary
    wins = dftr["r_multiple"] > 0
    win_rate = float(wins.mean() * 100.0)
    avgR = float(dftr["r_multiple"].mean())
    medR = float(dftr["r_multiple"].median())
    bestR = float(dftr["r_multiple"].max())
    worstR = float(dftr["r_multiple"].min())
    gains = dftr.loc[dftr["r_multiple"] > 0, "r_multiple"].sum()
    losses = -dftr.loc[dftr["r_multiple"] < 0, "r_multiple"].sum()
    pf = (gains / losses) if losses > 0 else float("inf")

    print("\n=== Backtest Summary ===")
    print(f"Trades: {len(dftr)}")
    print(f"Win%:   {win_rate:.1f}%")
    print(f"Avg R:  {avgR:.3f} | Med R: {medR:.3f}")
    print(f"Best/Worst R: {bestR:.3f} / {worstR:.3f}")
    print(f"PF:     {pf if pf!=float('inf') else 'inf'}")

    pair_stats = dftr.groupby("pair")["r_multiple"].agg(
        n="count", win=lambda s: (s>0).mean()*100.0, avg="mean", med="median", sum="sum"
    ).reset_index().sort_values("avg", ascending=False)
    pair_csv = out_csv.replace(".csv", "_pairs.csv")
    pair_stats.to_csv(pair_csv, index=False)
    print(f"[output] wrote per-pair summary -> {pair_csv}")

# ================= CLI =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    ap.add_argument("--universe", default="top100", choices=["top100","volume"])
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--gate", default="on", choices=["on","off"])
    ap.add_argument("--out", default="trades.csv")
    ap.add_argument("--fee_bps", type=float, default=8.0)
    ap.add_argument("--slip_bps", type=float, default=10.0)
    ap.add_argument("--max_hold_bars", type=int, default=60)
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f) or {}

    backtest(cfg,
             universe=args.universe,
             tf=args.tf,
             start=args.start,
             end=args.end,
             gate_on=(args.gate.lower()=="on"),
             out_csv=args.out,
             fee_bps=args.fee_bps,
             slip_bps=args.slip_bps,
             max_hold_bars=args.max_hold_bars)

if __name__ == "__main__":
    main()
