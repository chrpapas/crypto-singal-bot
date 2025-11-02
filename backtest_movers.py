#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest (Movers-only) for MEXC using legacy mover signal:
  - Universe from CoinMarketCap Movers (24h % change + volume + age filters) -> map to MEXC /USDT pairs
  - Entry rule on chosen tf (1h or 4h):
      EMA20 > EMA50, MACD line > signal, RSI in [55, 80],
      close > max(high, lookback=30), volume >= SMA(20)
  - Stop = min(low of last 10 bars) at signal bar
  - Target (t1) = +5%
  - Sizing = 1% of equity risk per trade (configurable)
  - Slippage + fees applied (configurable)
  - One position per symbol at a time; many concurrent allowed
  - Outputs: performance table + trades CSV + equity curve CSV

Environment:
  CMC_API_KEY  (optional; if missing we fallback to MEXC top USDT-volume pairs
                and pick Movers approximately via exchange data momentum)

Run:
  python3 backtest_movers.py --start 2025-07-01 --end 2025-10-29 --tf 4h
"""

import os, time, argparse, math, json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import requests
import ccxt

# --------------- INDICATORS ---------------
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100 / (1 + rs))
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow)
    line = f - s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# --------------- MEXC CLIENT ---------------
class Mexc:
    def __init__(self):
        self.ex = ccxt.mexc({"enableRateLimit": True})
        self._markets = None
    def load_markets(self):
        if self._markets is None:
            try: self._markets = self.ex.load_markets()
            except Exception: self._markets = {}
        return self._markets
    def has_pair(self, pair: str) -> bool:
        mkts = self.load_markets() or {}
        if pair in mkts: return True
        syms = getattr(self.ex, "symbols", None) or []
        return pair in syms
    def ohlcv(self, pair: str, tf: str, since_ms: Optional[int]=None, limit: int=2000) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(pair, timeframe=tf, since=since_ms, limit=limit)
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

# --------------- CMC HELPERS ---------------
def fetch_cmc_listings(limit=500, api_key: Optional[str]=None) -> List[dict]:
    k = api_key or os.environ.get("CMC_API_KEY","").strip()
    if not k: return []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": k}
    params = {"limit": limit, "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print("[cmc] listings error:", e)
        return []

def cmc_movers_symbols(min_change_24h=15.0, min_volume_usd=5_000_000, max_age_days=365, api_key=None) -> List[str]:
    data = fetch_cmc_listings(limit=500, api_key=api_key)
    if not data: return []
    out=[]; now=pd.Timestamp.utcnow()
    for it in data:
        sym=it.get("symbol","").upper()
        q=it.get("quote",{}).get("USD",{})
        ch=float(q.get("percent_change_24h") or 0)
        vol=float(q.get("volume_24h") or 0)
        date_added=pd.to_datetime(it.get("date_added", now.isoformat()), utc=True)
        if ((now - date_added).days) > max_age_days: continue
        if ch>=min_change_24h and vol>=min_volume_usd:
            out.append(sym)
    return out

# --------------- UNIVERSE ---------------
STABLES = {"USDT","USDC","FDUSD","TUSD","DAI","USDD","USDP","PYUSD","GUSD","USD1","USDE","EURS","EURT","XAUT","PAXG"}

def filter_pairs_on_mexc(client: Mexc, symbols: List[str], quote="USDT") -> List[str]:
    client.load_markets()
    pairs=[]
    for s in symbols:
        p=f"{s}/{quote}"
        if client.has_pair(p) and s not in STABLES:
            pairs.append(p)
    return pairs

def mexc_top_usdt_volume_pairs(client: Mexc, max_pairs=60, min_qv=2_000_000) -> List[str]:
    try:
        t = client.ex.fetch_tickers()
    except Exception as e:
        print("[fallback] fetch_tickers err:", e); return []
    items=[]
    for sym, rec in t.items():
        if "/USDT" not in sym: continue
        base = sym.split("/")[0].upper()
        if base in STABLES: continue
        qv = rec.get("quoteVolume")
        if qv is None:
            base_v = rec.get("baseVolume") or 0
            last = rec.get("last") or rec.get("close") or 0
            qv = float(base_v) * float(last)
        if float(qv or 0) >= float(min_qv):
            items.append((sym, float(qv)))
    items.sort(key=lambda x: x[1], reverse=True)
    out=[s for s,_ in items[:max_pairs]]
    print(f"[universe-fallback] picked {len(out)} pairs by USDT volume (min_qv={min_qv})")
    return out

# --------------- SIGNAL ---------------
def legacy_mover_signal_row(df: pd.DataFrame, i: int) -> Optional[Dict[str,Any]]:
    """Evaluate signal on row i using data up to i (inclusive)."""
    if i < 80: return None
    sl = slice(0, i+1)
    close = df["close"].iloc[sl]
    vol   = df["volume"].iloc[sl]
    e20 = ema(close, 20); e50 = ema(close, 50)
    mac_line, mac_sig, _ = macd(close)
    r = rsi(close,14)
    vS = sma(vol, 20)

    last_close = close.iloc[-1]
    last_volS = vS.iloc[-1] if not np.isnan(vS.iloc[-1]) else vol.iloc[-1]
    hl = df["high"].iloc[max(0, i-30):i].max()  # last 30 highs, excluding bar i

    aligned = (e20.iloc[-1] > e50.iloc[-1]) and (mac_line.iloc[-1] > mac_sig.iloc[-1]) and (55 <= r.iloc[-1] <= 80)
    breakout = (last_close > hl) and (vol.iloc[-1] >= last_volS)
    if not (aligned and breakout): return None

    stop = float(df["low"].iloc[max(0, i-9):i+1].min())  # min low of last 10 bars incl i
    return {
        "entry": float(last_close),
        "stop": float(stop),
        "level": float(hl),
        "event_ts": df.index[i]
    }

# --------------- BACKTEST ---------------
def simulate_pair(df: pd.DataFrame, pair: str, risk_pct=1.0, fee_bps=20, slip_bps=5) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Returns (trades, equity_points). trades: list with per-trade stats.
    equity_points: [{"ts":..., "equity":...}] using trade-to-trade equity (no mark-to-market).
    """
    if df.empty or len(df)<120: return [], []
    trades=[]; equity_curve=[]
    equity=10000.0  # start capital
    in_pos=False
    entry=None; stop=None; t1=None; qty=None; entry_ts=None

    def bps(x): return x/10000.0

    for i in range(len(df)-1):  # use next bar for fills
        row_ts = df.index[i]
        nxt = df.iloc[i+1]  # next bar for fills
        # if not in a position, check for signal on this bar
        if not in_pos:
            sig = legacy_mover_signal_row(df, i)
            if sig:
                ent = sig["entry"]
                stp = sig["stop"]
                if ent <= stp:  # guard
                    continue
                per_unit_risk = ent - stp
                risk_amt = equity * (risk_pct/100.0)
                q = risk_amt / per_unit_risk
                if q <= 0: continue
                # fill at next bar open + slippage up
                fill = float(nxt["open"]) * (1 + bps(slip_bps))
                # round-trip fees will be applied on exit (simplify) + entry one-way now
                fee_in = fill * q * (fee_bps/10000.0)
                cost = fill*q + fee_in
                if cost <= 0: continue
                in_pos=True
                entry=fill; stop=stp; t1=entry*1.05; qty=q; entry_ts=df.index[i+1]
        else:
            # manage position on next bar (nxt)
            lo=float(nxt["low"]); hi=float(nxt["high"])
            exit_reason=None; exit_px=None
            if hi >= t1:
                exit_reason="t1"
                exit_px=t1 * (1 - bps(slip_bps))  # positive slippage down a bit
            elif lo <= stop:
                exit_reason="stop"
                exit_px=stop * (1 - bps(slip_bps))  # slippage worsens the loss
            # exit on bar close if last bar (safety)
            if (i+1)==(len(df)-1) and exit_reason is None:
                exit_reason="close"
                exit_px=float(nxt["close"]) * (1 - bps(slip_bps))

            if exit_reason:
                # fees: one-way (entry) already taken via cost; apply exit fees here
                fee_out = exit_px * qty * (fee_bps/10000.0)
                pnl = (exit_px - entry) * qty - fee_out
                r_mult = (exit_px - entry) / max(1e-12, (entry - stop))
                equity += pnl
                trades.append({
                    "pair": pair,
                    "entry_ts": entry_ts.isoformat(),
                    "exit_ts": df.index[i+1].isoformat(),
                    "entry": round(entry, 8),
                    "stop": round(stop, 8),
                    "t1": round(t1, 8),
                    "exit": round(exit_px, 8),
                    "qty": round(qty, 6),
                    "pnl_usd": round(pnl, 2),
                    "r_multiple": round(r_mult, 3),
                    "outcome": exit_reason
                })
                equity_curve.append({"ts": df.index[i+1], "equity": equity})
                # reset
                in_pos=False; entry=None; stop=None; t1=None; qty=None; entry_ts=None
    return trades, equity_curve

def summarize(trades: List[Dict[str,Any]]):
    if not trades:
        print("No signals generated in the selected window.")
        return
    df = pd.DataFrame(trades)
    wins = df[df["r_multiple"]>0]
    losses = df[df["r_multiple"]<=0]
    win_rate = (len(wins)/len(df))*100.0 if len(df)>0 else 0.0
    avgR = df["r_multiple"].mean()
    pf = wins["r_multiple"].sum() / abs(losses["r_multiple"].sum()) if len(losses)>0 else float("inf")
    best = df["r_multiple"].max() if not df.empty else 0
    worst = df["r_multiple"].min() if not df.empty else 0
    dur = (pd.to_datetime(df["exit_ts"])-pd.to_datetime(df["entry_ts"])).dt.total_seconds()/3600.0 if not df.empty else []
    avg_h = float(dur.mean()) if len(dur)>0 else 0.0
    print("\n=== Performance (Movers-only) ===")
    print(f"Trades: {len(df)} | Win%: {win_rate:.1f}% | Avg R: {avgR:.2f} | PF: {pf:.2f} | Best/Worst R: {best:.2f}/{worst:.2f} | Avg Hold: {avg_h:.1f}h")

def max_drawdown(equity_points: List[Dict[str,Any]]) -> float:
    if not equity_points: return 0.0
    s = pd.Series([p["equity"] for p in equity_points])
    roll_max = s.cummax()
    dd = (s - roll_max) / roll_max
    return float(dd.min()*100.0)

# --------------- RUN ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="UTC end date (YYYY-MM-DD)")
    ap.add_argument("--tf", default="4h", choices=["1h","4h"], help="Timeframe")
    ap.add_argument("--risk_pct", type=float, default=1.0)
    ap.add_argument("--fee_bps", type=float, default=20.0, help="Round-trip bps (each side applied separately)")
    ap.add_argument("--slip_bps", type=float, default=5.0)
    ap.add_argument("--limit_pairs", type=int, default=80)
    ap.add_argument("--min_change_24h", type=float, default=15.0)
    ap.add_argument("--min_vol_usd_24h", type=float, default=5_000_000)
    ap.add_argument("--max_age_days", type=int, default=365)
    ap.add_argument("--fallback", action="store_true", help="Use MEXC volume universe approx if no CMC key")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end, tz="UTC")
    tf = args.tf

    mexc = Mexc()

    # Universe: CMC Movers -> MEXC pairs
    cmc_key = os.environ.get("CMC_API_KEY","").strip()
    if cmc_key:
        movers_syms = cmc_movers_symbols(
            min_change_24h=args.min_change_24h,
            min_volume_usd=args.min_vol_usd_24h,
            max_age_days=args.max_age_days,
            api_key=cmc_key
        )
        movers_syms = movers_syms[:max(args.limit_pairs, 1)]
        pairs = filter_pairs_on_mexc(mexc, movers_syms, quote="USDT")
        print(f"[universe] CMC movers {len(movers_syms)} -> MEXC pairs {len(pairs)} (limited to {args.limit_pairs})")
    else:
        if not args.fallback:
            print("No CMC_API_KEY and --fallback not set. Exiting.")
            return
        pairs = mexc_top_usdt_volume_pairs(mexc, max_pairs=args.limit_pairs, min_qv=2_000_000)

    if not pairs:
        print("No tradable pairs in universe.")
        return

    # Fetch & backtest
    all_trades=[]; all_eq=[]
    tf_ms = {"1h": 60*60*1000, "4h": 4*60*60*1000}[tf]
    since_ms = int(start.value/1e6) - 10*tf_ms  # little buffer

    for p in pairs:
        # throttle gently to avoid rate limits
        time.sleep(0.15)
        df = mexc.ohlcv(p, tf, since_ms=since_ms, limit=5000)
        if df.empty: continue
        # slice to window, tz-aware
        df = df.loc[(df.index >= start) & (df.index <= end)].copy()
        if len(df) < 120: continue
        t, eq = simulate_pair(df, p, risk_pct=args.risk_pct, fee_bps=args.fee_bps, slip_bps=args.slip_bps)
        all_trades.extend([{**x, "pair": p} for x in t])
        # merge equity as stepwise curve
        all_eq.extend(eq)

    # Results
    summarize(all_trades)
    if all_eq:
        # build full equity curve over time
        eq_df = pd.DataFrame(all_eq).sort_values("ts")
        # start at 10k
        base = 10000.0
        if not eq_df.empty:
            # ensure we start with base point
            eq_df = eq_df.drop_duplicates("ts", keep="last")
            dd = max_drawdown(eq_df.to_dict("records"))
            print(f"Max Drawdown: {dd:.2f}%")

        # save files
        trades_path = "movers_trades.csv"
        curve_path = "movers_equity_curve.csv"
        pd.DataFrame(all_trades).to_csv(trades_path, index=False)
        eq_df.to_csv(curve_path, index=False)
        print(f"\nSaved: {trades_path} and {curve_path}")

if __name__ == "__main__":
    main()
