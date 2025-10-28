#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# backtest_mexc_bot.py — bar-by-bar backtester for mexc_trader_bot.py
# Adds ETH regime gating for bullish signals in the Top100 universe.

import argparse, os, yaml
import pandas as pd
import numpy as np
import mexc_trader_bot as bot  # reuse your live logic

def _simulate_path(df, side, entry, stop, t1, t2):
    """Walk forward bar-by-bar; priority: Stop first, then T2, then T1 (for shorts: reverse)."""
    risk = abs(entry - stop)
    if side == "short" and (t1 is None or t2 is None):
        # default 1R / 2R targets for shorts (bearish signals)
        t1, t2 = entry - 1*risk, entry - 2*risk
    if side == "long" and (t1 is None or t2 is None):
        # default 1R / 2R if bullish signal provided None
        t1, t2 = entry + 1*risk, entry + 2*risk

    for i, r in enumerate(df.itertuples(index=False), start=1):
        lo, hi = float(r.low), float(r.high)
        if side == "long":
            if lo <= stop: return ("stop", stop, i)
            if hi >= t2:   return ("t2", t2, i)
            if hi >= t1:   return ("t1", t1, i)
        else:
            if hi >= stop: return ("stop", stop, i)
            if lo <= t2:   return ("t2", t2, i)
            if lo <= t1:   return ("t1", t1, i)
    return ("open", float(df["close"].iloc[-1]), len(df))

def _r_multiple(side, entry, stop, exit_px):
    r = abs(entry - stop) or 1e-12
    return (exit_px - entry)/r if side=="long" else (entry - exit_px)/r

def _eth_regime_ok(hist_eth: pd.DataFrame, eth_cfg: dict) -> bool:
    """
    ETH regime gate for bullish signals (on Top100):
      - EMA20>EMA50 if require_ema_stack
      - MACD line > signal if require_macd_bull
      - RSI >= rsi_min
      - Optional: last close > prior swing high (breakout_lookback) if require_above_reghigh
    """
    if hist_eth is None or len(hist_eth) < 80:
        return False
    ok = bot.tf_bull_ok(
        hist_eth,
        require_ema_stack=bool(eth_cfg.get("require_ema_stack", True)),
        require_macd_bull=bool(eth_cfg.get("require_macd_bull", True)),
        min_rsi=int(eth_cfg.get("rsi_min", 52)),
    )
    if not ok:
        return False
    if eth_cfg.get("require_above_reghigh", False):
        lb = int(eth_cfg.get("breakout_lookback", 30))
        if len(hist_eth) < lb + 5:
            return False
        last = hist_eth.iloc[-1]
        reghigh = float(hist_eth["high"].iloc[-(lb+1):-1].max())
        if not (float(last["close"]) > reghigh):
            return False
    return True

def backtest(cfg, universe_mode, start, end, tf, capital, risk_pct, include_bear):
    client = bot.ExClient()
    extra_stables = cfg.get("filters", {}).get("extra_stables", [])

    # Universe (Top100 or Movers). ETH regime gate only applies to Top100.
    if universe_mode=="top100":
        syms = bot.cmc_top100_symbols(cfg)
        pairs = bot.filter_pairs_on_mexc(client, syms, "USDT", extra_stables)
        if not pairs:
            # fallback to top USDT-volume if CMC mapping is empty
            pairs = bot.mexc_top_usdt_volume_pairs(client, max_pairs=60, min_usd_vol=2_000_000, extra_stables=extra_stables)
    else:
        syms = bot.cmc_movers_symbols(cfg)
        pairs = bot.filter_pairs_on_mexc(client, syms, "USDT", extra_stables)

    print(f"[universe] {len(pairs)} pairs ({universe_mode}) | tf={tf}")

    # Params + detectors
    dayP   = bot.DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP   = bot.TrendParams(**cfg.get("trend_trade_params", {}))
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    bear_cfg = cfg.get("bearish_signals", {"enabled": True})
    eth_cfg = cfg.get("eth_regime", {
        "enabled": True,
        "require_ema_stack": True,
        "require_macd_bull": True,
        "rsi_min": 52,
        "breakout_lookback": 30,
        "require_above_reghigh": False
    })

    # Bullish & bearish signal callables (reuse live logic)
    bull = {
        "1h": lambda d: bot.day_signal(d, dayP, sd_cfg, None),
        "4h": lambda d: bot.swing_signal(d, swingP, sd_cfg, None),
        "1d": lambda d: bot.trend_signal(d, trnP, sd_cfg, None)
    }[tf]
    bear = {
        "1h": lambda d: bot.day_bearish_signal(d, bear_cfg),
        "4h": lambda d: bot.swing_bearish_signal(d, bear_cfg),
        "1d": lambda d: bot.trend_bearish_signal(d, bear_cfg)
    }[tf]

    # Preload ETH/USDT for the same timeframe (for gating on Top100)
    eth_df = None
    if universe_mode == "top100" and eth_cfg.get("enabled", True):
        try:
            lim = 1200 if tf in ("1h","4h") else 800
            eth_df = client.ohlcv("ETH/USDT", tf, lim)
            # Trim to requested range early (keeps index aligned)
            eth_df = eth_df.loc[pd.to_datetime(start): pd.to_datetime(end)]
        except Exception:
            eth_df = None
            print("[eth-regime] failed to fetch ETH/USDT; gate will be OFF.")

    trades=[]; eq=capital

    for pair in pairs:
        # Load pair OHLCV
        try:
            limit = 1200 if tf in ("1h","4h") else 800
            df_all = client.ohlcv(pair, tf, limit)
        except Exception:
            continue
        if df_all.empty: 
            continue

        df = df_all.loc[pd.to_datetime(start): pd.to_datetime(end)]
        if len(df) < 100:
            continue

        # Walk forward
        for i in range(60, len(df)-1):
            hist = df.iloc[:i]
            ts = df.index[i]

            # ---------- Bullish (ETH-gated only for Top100) ----------
            s = bull(hist)
            if s:
                eth_gate_ok = True
                if universe_mode == "top100" and eth_cfg.get("enabled", True) and eth_df is not None and not eth_df.empty:
                    # Use ETH history up to the same timestamp (align by index)
                    hist_eth = eth_df.loc[:ts]
                    eth_gate_ok = _eth_regime_ok(hist_eth, eth_cfg)
                if eth_gate_ok:
                    entry = float(hist["close"].iloc[-1]); stop=float(s["stop"])
                    t1=s.get("t1"); t2=s.get("t2")
                    out, px, held = _simulate_path(df.iloc[i+1:], "long", entry, stop, t1, t2)
                    R = _r_multiple("long", entry, stop, px)
                    pnl = (eq * risk_pct/100.0) * R; eq += pnl
                    trades.append(dict(pair=pair,side="long",tf=tf,ts=str(ts),
                                       entry=entry,stop=stop,exit=px,out=out,bars=held,R=R,pnl=pnl,equity=eq))
            # ---------- Bearish (no ETH gate; you asked to always produce these) ----------
            if include_bear and bear_cfg.get("enabled", True):
                sb = bear(hist)
                if sb:
                    entry = float(hist["close"].iloc[-1])
                    stop  = float(sb.get("stop", hist["high"].iloc[-1]))
                    out, px, held = _simulate_path(df.iloc[i+1:], "short", entry, stop, None, None)
                    R = _r_multiple("short", entry, stop, px)
                    pnl = (eq * risk_pct/100.0) * R; eq += pnl
                    trades.append(dict(pair=pair,side="short",tf=tf,ts=str(ts),
                                       entry=entry,stop=stop,exit=px,out=out,bars=held,R=R,pnl=pnl,equity=eq))

    if not trades:
        print("No trades produced.")
        return

    # ----- Stats -----
    t = pd.DataFrame(trades)
    def stats(d):
        if d.empty: return (0,0,0,0,0,0)
        wins = (d["R"]>0).mean()*100.0
        gains = d.loc[d.R>0,"R"].sum()
        losses = -d.loc[d.R<0,"R"].sum()
        pf = gains / (losses if losses>0 else 1e-12)
        return (len(d), wins, d["R"].mean(), pf, d["R"].max(), d["R"].min())

    n, w, avgR, pf, bestR, worstR = stats(t)
    ln, lw, lavgR, lpf, _, _ = stats(t[t.side=="long"])
    sn, sw, savgR, spf, _, _ = stats(t[t.side=="short"])

    print("\n=== Backtest Summary ===")
    print(f"All   : N {n} | Win% {w:.1f}% | AvgR {avgR:.2f} | PF {pf:.2f} | Best/Worst {bestR:.2f}/{worstR:.2f}")
    print(f"Longs : N {ln} | Win% {lw:.1f}% | AvgR {lavgR:.2f} | PF {lpf:.2f}")
    print(f"Shorts: N {sn} | Win% {sw:.1f}% | AvgR {savgR:.2f} | PF {spf:.2f}")
    print(f"End equity: {t['equity'].iloc[-1]:.2f} (start {capital:.2f})")

    t.to_csv("backtest_trades.csv", index=False)
    print("Saved trades to backtest_trades.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    ap.add_argument("--universe", default="top100", choices=["top100","movers"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tf", default="1h", choices=["1h","4h","1d"])
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--include-bear", default="true")
    args = ap.parse_args()

    with open(args.config,"r") as f: cfg = yaml.safe_load(f)

    # env expansion ${VAR}
    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o
    cfg = expand_env(cfg)

    backtest(cfg,
             universe_mode=args.universe,
             start=args.start, end=args.end, tf=args.tf,
             capital=float(args.capital), risk_pct=float(args.risk),
             include_bear=(str(args.include_bear).lower() in ("1","true","yes","y")))

if __name__ == "__main__":
    main()
