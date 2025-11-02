#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtester for MEXC signal logic with ETH-gate and historical Movers timeline (built from 5m MEXC data),
now with trade simulation and performance metrics.

Run examples:
  python3 backtest_mexc_bot.py --config mexc_trader_bot_config.yml --universe top100 --tf 4h --start 2025-07-01 --end 2025-10-29
  python3 backtest_mexc_bot.py --config mexc_trader_bot_config.yml --universe movers  --tf 4h --start 2025-07-01 --end 2025-10-29

Notes
- Uses ccxt.mexc for OHLCV (spot).
- Movers are reconstructed historically from 5m bars (rolling 24h % and 24h USD volume).
- ETH gate is controlled by config (eth_gate.enabled). If disabled in YAML, it’s OFF even if CLI “gate_on” default is true.
- Exits:
    * 50% at +1R, then stop at breakeven for the remainder
    * Chandelier trail for remainder: highest close since TP1 - K * ATR(N), default N=20, K=3.0
    * Optional time stop after M bars (default 30 bars on TF)
- Conservative intrabar assumption: if a bar hits both stop and target(s), stop wins.
"""

import argparse, os, yaml, math, time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import ccxt

# ------------------------ TA helpers ------------------------
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff(); up = d.clip(lower=0).rolling(length).mean(); dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9); return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow); line = f - s; sig = line.ewm(span=signal, adjust=False).mean(); hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder ATR via EWM."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ------------------------ Signal defs (same logic as bot) ------------------------
def day_signal(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    look = int(params.get("lookback_high", 30)); voln = int(params.get("vol_sma", 30))
    if len(df) < max(look, voln) + 5: return None
    volS = sma(df["volume"], voln)
    r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()
    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) \
                    and (params.get("rsi_min", 52) <= r.iloc[-1] <= params.get("rsi_max", 78))
    retest_edge = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok   = retest_edge and (last["volume"] > 0.8*(volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"])) \
                    and (r.iloc[-1] >= params.get("rsi_min", 52))
    if not (breakout_ok or retrec_ok): return None
    entry = float(last["close"]); stop = float(min(df["low"].iloc[-10:]))
    return {"type": "day", "entry": entry, "stop": stop, "t1": round(entry*1.05, 6), "t2": round(entry*1.10, 6),
            "level": float(highlvl), "note": "Breakout" if breakout_ok else "Retest-Reclaim"}

def swing_signal(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    need = max(params.get('ema100',100), params.get('vol_sma',20), params.get('breakout_lookback',34)) + 5
    if len(df) < need: return None
    df = df.copy()
    df['ema20']=ema(df['close'],params.get('ema20',20)); df['ema50']=ema(df['close'],params.get('ema50',50))
    df['ema100']=ema(df['close'],params.get('ema100',100)); df['volS']=sma(df['volume'],params.get('vol_sma',20))
    r=rsi(df['close'],14); last=df.iloc[-1]
    aligned=(last['ema20']>last['ema50']>last['ema100']) and (r.iloc[-1] >= params.get('rsi_min',50))
    within=abs((last['close']-last['ema20'])/max(1e-12,last['ema20'])*100)<=params.get('pullback_pct_max',10.0)
    bounce=last['close'] > df['close'].iloc[-2]
    hl=df['high'].iloc[-(params.get('breakout_lookback',34)+1):-1].max()
    breakout=(last['close']>hl) and (last['volume']> (df['volS'].iloc[-1] if not np.isnan(df['volS'].iloc[-1]) else last['volume']))
    if not (aligned and ((within and bounce) or breakout)): return None
    entry=float(last['close']); stop=float(min(df['low'].iloc[-10:]))
    return {"type":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
            "level":float(hl),"note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout"}

def trend_signal(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    need = max(params.get("ema100",100), params.get("vol_sma",20), params.get("breakout_lookback",55)) + 5
    if len(df) < need: return None
    df = df.copy()
    df["ema20"]=ema(df["close"],params.get("ema20",20)); df["ema50"]=ema(df["close"],params.get("ema50",50))
    df["ema100"]=ema(df["close"],params.get("ema100",100)); df["volS"]=sma(df["volume"],params.get("vol_sma",20))
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1] >= params.get("rsi_min",50))
    within=abs((last["close"]-last["ema20"])/max(1e-12,last["ema20"])*100)<=params.get("pullback_pct_max",10.0)
    bounce= last["close"] > df["close"].iloc[-2]
    hl=df["high"].iloc[-(params.get("breakout_lookback",55)+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]> (df["volS"].iloc[-1] if not np.isnan(df["volS"].iloc[-1]) else last["volume"]))
    if not (aligned and ((within and bounce) or breakout)): return None
    entry=float(last["close"]); stop=float(min(df["low"].iloc[-10:]))
    return {"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
            "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout"}

# ------------------------ ETH gate ------------------------
def eth_gate_ok(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """
    Returns a boolean series gate_ok indexed like df:
      gate_ok = (ema20 > ema50 if required) and (RSI >= min_rsi if set)
    """
    if df is None or df.empty:
        return pd.Series(False, index=pd.Index([], dtype='datetime64[ns, UTC]'))
    e20 = ema(df["close"], 20); e50 = ema(df["close"], 50); r = rsi(df["close"],14)
    ok = pd.Series(True, index=df.index)
    if cfg.get("require_ema_stack", True):
        ok &= (e20 > e50)
    if cfg.get("min_rsi", None) is not None:
        ok &= (r >= float(cfg["min_rsi"]))
    return ok.rename("gate_ok")

# ------------------------ ccxt helpers ------------------------
def ccxt_timeframe_ms(tf: str) -> int:
    m = {"1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000,"30m":1_800_000,
         "1h":3_600_000,"2h":7_200_000,"4h":14_400_000,"1d":86_400_000}
    return m[tf]

def fetch_ohlcv_tz_aware(ex: ccxt.Exchange, symbol: str, timeframe: str,
                         start: pd.Timestamp, end: pd.Timestamp, limit=1000) -> pd.DataFrame:
    """
    Pulls OHLCV iteratively between [start,end], returns UTC tz-aware DataFrame indexed by ts (UTC).
    """
    tf_ms = ccxt_timeframe_ms(timeframe)
    since = int(start.timestamp()*1000)
    out = []
    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        out.extend(candles)
        last_ts = candles[-1][0]
        next_since = last_ts + tf_ms
        if next_since >= int(end.timestamp()*1000):
            break
        if next_since <= since:
            break
        since = next_since
        time.sleep(ex.rateLimit/1000.0 if getattr(ex, "rateLimit", None) else 0.2)
    if not out:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df

# ------------------------ Movers timeline (5m) ------------------------
def add_rolling_5m_metrics(df_5m: pd.DataFrame) -> pd.DataFrame:
    df = df_5m.copy()
    window = 288  # 24h worth of 5m bars
    df["ret_24h"] = df["close"].pct_change(window)
    df["vol_usd_24h"] = (df["volume"] * df["close"]).rolling(window, min_periods=window).sum()
    return df

def build_movers_timeline_5m(df5_by_pair: Dict[str, pd.DataFrame],
                             min_change_24h_pct: float,
                             min_vol_usd_24h: float,
                             limit: int) -> Dict[pd.Timestamp, List[str]]:
    all_ts = sorted(set().union(*[df.index for df in df5_by_pair.values()])) if df5_by_pair else []
    timeline = {}
    for ts in all_ts:
        picks = []
        for pair, df in df5_by_pair.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            ret = float(row.get("ret_24h") or 0.0)
            volusd = float(row.get("vol_usd_24h") or 0.0)
            if not (np.isfinite(ret) and np.isfinite(volusd)):
                continue
            if (ret*100.0) >= min_change_24h_pct and volusd >= min_vol_usd_24h:
                picks.append((pair, ret, volusd))
        picks.sort(key=lambda x: x[1], reverse=True)
        timeline[ts] = [p for p,_,_ in picks[:limit]]
    return timeline

def last_snapshot_leq(sorted_keys: List[pd.Timestamp], ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    import bisect
    i = bisect.bisect_right(sorted_keys, ts) - 1
    return sorted_keys[i] if i >= 0 else None

# ------------------------ Universe builders ------------------------
DEFAULT_STABLES = {"USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT","USD1","USDE","USDY","USDP","SUSD","EURS","EURT","PYUSD"}

def is_stable_base(base: str, extra: List[str]) -> bool:
    b = base.upper().replace("3L","").replace("3S","").replace("5L","").replace("5S","")
    return (b in DEFAULT_STABLES) or (b in {e.upper() for e in (extra or [])})

def has_pair(ex: ccxt.Exchange, pair: str) -> bool:
    mkts = getattr(ex, "markets", None)
    if mkts is None:
        try: ex.load_markets()
        except Exception: return False
        mkts = ex.markets
    return pair in mkts or pair in getattr(ex, "symbols", [])

def mexc_volume_universe(ex: ccxt.Exchange, *, quote="USDT", max_pairs=80, min_usd_vol=2_000_000, extra_stables=None) -> List[str]:
    extra_stables = extra_stables or []
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:
        print("[universe] fetch_tickers err:", e)
        return []
    rows=[]
    for sym,t in tickers.items():
        if f"/{quote}" not in sym: continue
        base,_ = sym.split("/")
        if is_stable_base(base, extra_stables): continue
        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume") or 0
            last = t.get("last") or t.get("close") or 0
            qv = base_v * last
        try: qv = float(qv or 0)
        except: qv = 0.0
        if qv >= float(min_usd_vol):
            rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s,_ in rows[:max_pairs]]
    print(f"[universe] volume universe -> {len(pairs)} pairs >= ${min_usd_vol:,.0f} 24h quote vol")
    return pairs

def cmc_top100_symbols(cfg: Dict[str,Any]) -> List[str]:
    key = os.environ.get("CMC_API_KEY") or cfg.get("movers",{}).get("cmc_api_key","")
    if not key:
        return []
    import requests
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    try:
        r = requests.get(url, headers={"X-CMC_PRO_API_KEY": key}, params={"limit":140,"convert":"USD"}, timeout=15)
        data = r.json().get("data", [])
        data.sort(key=lambda x: x.get("quote",{}).get("USD",{}).get("market_cap",0), reverse=True)
        return [it["symbol"].upper() for it in data[:110] if "symbol" in it]
    except Exception as e:
        print("[cmc] error:", e); return []

def map_to_mexc_pairs(ex: ccxt.Exchange, syms: List[str], quote="USDT", extra_stables=None) -> List[str]:
    extra_stables = extra_stables or []
    out=[]
    for s in syms:
        p = f"{s}/{quote}"
        if not has_pair(ex, p): continue
        base,_ = p.split("/")
        if is_stable_base(base, extra_stables): continue
        out.append(p)
    return out

# ------------------------ Perf helpers (exits & metrics) ------------------------
def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series - cummax)
    return float(dd.min())

def simulate_trade_standard_exit(
    df: pd.DataFrame,
    ts_entry: pd.Timestamp,
    entry: float,
    stop: float,
    t1: float,
    t2: float,
    *,
    atr_col: str = "ATR",
    atr_len: int = 20,
    atr_k: float = 3.0,
    time_stop_bars: int = 30
) -> Tuple[float, str, pd.Timestamp]:
    """
    Standard exit:
      - 50% at +1R (t1 can be used but we explicitly enforce +1R target for partial)
      - Move stop on remainder to breakeven (entry)
      - Trail remainder using chandelier: highest_close_since_TP1 - K*ATR(N)
      - Optional time stop after M bars closes position at market
    Conservative intrabar ordering:
      - Before TP1: SL first, then targets
      - After TP1: BE first, then trail, then T2
    Returns: (R_multiple_total, exit_reason, ts_exit_last)
    """
    fwd = df.loc[df.index > ts_entry]
    if fwd.empty:
        return (0.0, "NO_DATA", ts_entry)

    R_unit = (entry - stop)
    if R_unit <= 0:
        return (0.0, "BAD_STOP", ts_entry)

    # Targets: enforce +1R for TP1 regardless of provided t1
    tp1_px = entry + R_unit        # +1R
    tp2_px = float(t2) if t2 and np.isfinite(t2) else entry + 2.0 * R_unit

    realized_R = 0.0
    have_tp1 = False
    bars_after_entry = 0
    highest_close_since_tp1 = -float("inf")
    last_ts_exit = ts_entry

    for ts, row in fwd.iterrows():
        bars_after_entry += 1
        hi = float(row["high"]); lo = float(row["low"]); cl = float(row["close"])
        atr_now = float(row.get(atr_col, np.nan))

        # --- BEFORE TP1: full position ---
        if not have_tp1:
            # conservative: SL first
            if lo <= stop:
                return (-1.0, "SL", ts)  # full loss
            # then check +1R / TP1 and TP2 (TP2 may be hit same bar)
            hit_tp1 = hi >= tp1_px
            hit_tp2 = hi >= tp2_px
            if hit_tp1:
                # book 50% at +1R
                realized_R += 0.5 * 1.0  # +0.5R
                have_tp1 = True
                last_ts_exit = ts
                # same-bar BE check (conservative: BE before TP2)
                if lo <= entry:
                    return (realized_R, "TP1_then_BE_samebar", ts)
                # same-bar TP2 check
                if hit_tp2:
                    realized_R += 0.5 * ((tp2_px - entry) / R_unit)  # + remainder to TP2
                    return (realized_R, "TP2_after_TP1_samebar", ts)
                # move on; now managing remainder
                highest_close_since_tp1 = max(highest_close_since_tp1, cl)
            else:
                # optional time stop before TP1
                if time_stop_bars and bars_after_entry >= time_stop_bars:
                    R_now = (cl - entry) / R_unit
                    return (R_now, "TIME_STOP_BEFORE_TP1", ts)
                continue

        # --- AFTER TP1: remainder with BE + chandelier trail ---
        # update highest close
        highest_close_since_tp1 = max(highest_close_since_tp1, cl)
        trail = None
        if np.isfinite(atr_now):
            trail = highest_close_since_tp1 - atr_k * atr_now

        # Order: BE first, then trail, then TP2
        if lo <= entry:
            return (realized_R, "BE_after_TP1", ts)

        if trail is not None and lo <= trail:
            return (realized_R, f"TRAIL_hit_{atr_len}x{atr_k}", ts)

        if hi >= tp2_px:
            realized_R += 0.5 * ((tp2_px - entry) / R_unit)
            return (realized_R, "TP2_after_TP1", ts)

        # optional time stop after TP1
        if time_stop_bars and bars_after_entry >= time_stop_bars:
            # close remainder at market
            rem_R = 0.5 * ((cl - entry) / R_unit)
            realized_R += rem_R
            return (realized_R, "TIME_STOP_AFTER_TP1", ts)

        last_ts_exit = ts

    # if never exited by end of data: close at last close
    last = fwd.iloc[-1]
    px = float(last["close"])
    if not have_tp1:
        R = (px - entry) / R_unit
        return (R, "EOD_BEFORE_TP1", fwd.index[-1])
    else:
        rem_R = 0.5 * ((px - entry) / R_unit)
        realized_R += rem_R
        return (realized_R, "EOD_AFTER_TP1", fwd.index[-1])

# ------------------------ Backtest core ------------------------
def backtest(cfg: Dict[str,Any], *, universe: str, tf: str, start: str, end: str):
    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc   = pd.Timestamp(end, tz="UTC")
    ex = ccxt.mexc({"enableRateLimit": True})

    # Universe
    extra_stables = cfg.get("filters",{}).get("extra_stables", [])
    pairs: List[str] = []
    if universe == "top100":
        syms = cmc_top100_symbols(cfg)
        if syms:
            pairs = map_to_mexc_pairs(ex, syms, "USDT", extra_stables)
            print(f"[universe] mapped top100 -> {len(pairs)} pairs")
        if not pairs:  # fallback
            pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=80, min_usd_vol=2_000_000, extra_stables=extra_stables)
    elif universe == "volume":
        pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=80, min_usd_vol=2_000_000, extra_stables=extra_stables)
    elif universe == "movers":
        # for movers we still need a candidate pool; take volume pool
        pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=120, min_usd_vol=2_000_000, extra_stables=extra_stables)
    else:
        print(f"[universe] unknown '{universe}', defaulting to volume")
        pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=80, min_usd_vol=2_000_000, extra_stables=extra_stables)

    print(f"[universe] {len(pairs)} pairs ({universe}) | tf={tf}")

    # ETH gate
    gate_cfg = cfg.get("eth_gate", {"enabled": True, "require_ema_stack": True, "min_rsi": 50})
    gate_on = bool(gate_cfg.get("enabled", True))
    eth_gate_series = pd.DataFrame()
    if gate_on:
        try:
            df_eth = fetch_ohlcv_tz_aware(ex, "ETH/USDT", tf, start_utc - pd.Timedelta(days=30), end_utc)
            if df_eth.empty:
                print("[eth-gate] series empty; gate will be OFF.")
                gate_on = False
            else:
                eth_gate_series = pd.DataFrame({
                    "close": df_eth["close"],
                    "gate_ok": eth_gate_ok(df_eth, gate_cfg)
                })
                last = eth_gate_series.iloc[-1]
                print(f"[eth-gate] {tf}:ON -> gate_ok(last)={bool(last['gate_ok'])}")
        except Exception as e:
            print("[eth-gate] failed to fetch ETH/USDT; gate will be OFF.")
            gate_on = False
    else:
        print(f"[eth-gate] {tf}:OFF (config disabled)")

    # Fetch main TF data for all pairs (+ ATR precompute)
    tf_df: Dict[str, pd.DataFrame] = {}
    warmup_days = 90
    for p in pairs:
        try:
            df = fetch_ohlcv_tz_aware(ex, p, tf, start_utc - pd.Timedelta(days=warmup_days), end_utc)
            if df is not None and not df.empty:
                df = df.copy()
                df["ATR"] = atr(df, n=20)
                tf_df[p] = df
        except Exception:
            pass
    if not tf_df:
        print("No data fetched on main TF; aborting.")
        return

    # Movers timeline (from 5m)
    mv_cfg = cfg.get("movers", {"min_change_24h":15.0,"min_volume_usd_24h":5_000_000.0,"limit":80})
    df5_by_pair: Dict[str, pd.DataFrame] = {}
    if universe == "movers":
        for p in pairs:
            try:
                df5 = fetch_ohlcv_tz_aware(ex, p, "5m", start_utc - pd.Timedelta(days=3), end_utc)
                if df5 is None or df5.empty:
                    continue
                df5 = add_rolling_5m_metrics(df5)
                df5_by_pair[p] = df5
            except Exception:
                pass
        movers_timeline = build_movers_timeline_5m(
            df5_by_pair,
            float(mv_cfg.get("min_change_24h", 15.0)),
            float(mv_cfg.get("min_volume_usd_24h", 5_000_000.0)),
            int(mv_cfg.get("limit", 80))
        )
        timeline_keys = sorted(movers_timeline.keys())
        print(f"[movers] timeline built with {len(timeline_keys)} snapshots")
    else:
        movers_timeline, timeline_keys = {}, []

    # Parameters
    dayP   = cfg.get("day_trade_params",   {"lookback_high":30, "vol_sma":30, "rsi_min":52, "rsi_max":78})
    swingP = cfg.get("swing_trade_params", {"ema20":20,"ema50":50,"ema100":100,"rsi_min":50,"pullback_pct_max":10.0,"vol_sma":20,"breakout_lookback":34})
    trendP = cfg.get("trend_trade_params", {"ema20":20,"ema50":50,"ema100":100,"rsi_min":50,"pullback_pct_max":10.0,"vol_sma":20,"breakout_lookback":55})

    # Backtest loop
    total = 0
    # (ts, pair, tf, typ, entry, stop, t1, t2)
    hits: List[Tuple[pd.Timestamp,str,str,str,float,float,float,float]] = []

    common_ts = sorted(set().union(*[df.index for df in tf_df.values()]))
    common_ts = [t for t in common_ts if (t >= start_utc and t <= end_utc)]
    if not common_ts:
        print("No bars in selected window after slicing.")
        return

    for ts in common_ts:
        if universe == "movers":
            snap = last_snapshot_leq(timeline_keys, ts)
            scan_pairs = movers_timeline.get(snap, []) if snap else []
        else:
            scan_pairs = pairs
        if not scan_pairs:
            continue

        # ETH gate check at ts
        gate_block = False
        if gate_on and (ts in eth_gate_series.index):
            gate_block = not bool(eth_gate_series.loc[ts, "gate_ok"])

        for pair in scan_pairs:
            df = tf_df.get(pair)
            if df is None or (ts not in df.index):
                continue
            dfx = df.loc[:ts]
            if len(dfx) < 60:
                continue

            sig = None
            for fn, par, name in ((day_signal, dayP, "day"), (swing_signal, swingP, "swing"), (trend_signal, trendP, "trend")):
                s = fn(dfx, par)
                if s:
                    s["type"] = name
                    sig = s
                    break
            if not sig:
                continue
            if gate_block:
                continue

            total += 1
            hits.append((
                ts, pair, tf, sig["type"],
                float(sig["entry"]), float(sig["stop"]),
                float(sig.get("t1", sig["entry"]*1.05)),
                float(sig.get("t2", sig["entry"]*1.10))
            ))

    # Report + Performance
    print(f"\n=== Backtest Summary ===")
    print(f"Bars tested: {len(common_ts)}  | Pairs: {len(pairs)}  | Universe: {universe.upper()}  | TF: {tf}")
    print(f"Signals generated: {len(hits)}")
    if hits:
        print("\nFirst 20 signals:")
        for ts, pair, tfv, typ, entry, stop, t1, t2 in hits[:20]:
            print(f"  {ts.isoformat()}  {pair} {tfv} {typ}  entry={entry:.6f} stop={stop:.6f}")

        # ===== Simulate trades (standard exit) & compute performance =====
        rows = []
        for ts, pair, tfv, typ, entry, stop, t1, t2 in hits:
            dfp = tf_df.get(pair)
            if dfp is None:
                continue
            R, reason, ts_exit = simulate_trade_standard_exit(
                dfp, ts, entry, stop, t1, t2,
                atr_col="ATR", atr_len=20, atr_k=3.0, time_stop_bars=30
            )
            rows.append({
                "ts_entry": ts, "pair": pair, "tf": tfv, "type": typ,
                "entry": entry, "stop": stop, "t1_used": entry + (entry - stop), "t2_used": t2,
                "R": R, "exit_reason": reason, "ts_exit": ts_exit
            })

        if not rows:
            print("\n[perf] No trades could be simulated (no forward bars).")
            return

        perf = pd.DataFrame(rows).sort_values("ts_entry").reset_index(drop=True)
        perf["cum_R"] = perf["R"].cumsum()

        # basic stats
        n = len(perf)
        wins = int((perf["R"] > 0).sum())
        losses = n - wins
        win_rate = (wins / n * 100.0) if n else 0.0
        avg_R = float(perf["R"].mean())
        exp_R = avg_R
        gross_win = float(perf.loc[perf["R"] > 0, "R"].sum())
        gross_loss = -float(perf.loc[perf["R"] < 0, "R"].sum())
        profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else float("inf")
        mdd_R = max_drawdown(perf["cum_R"])

        print("\n=== Performance (R multiples, standard exit: 50% @ +1R, BE, ATR(20)*3 trail) ===")
        print(f"Trades: {n} | Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.1f}%")
        print(f"Avg R/trade: {avg_R:.3f} | Expectancy: {exp_R:.3f} R")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Total R: {float(perf['R'].sum()):.2f} | Max Drawdown: {mdd_R:.2f} R")

        # Save CSVs
        out_dir = "./backtest_out"
        os.makedirs(out_dir, exist_ok=True)
        perf.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        perf[["ts_entry","cum_R"]].to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)
        print(f"[saved] {out_dir}/trades.csv  and  {out_dir}/equity_curve.csv")

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    ap.add_argument("--universe", choices=["top100","movers","volume"], default="top100")
    ap.add_argument("--tf", choices=["1h","4h","1d"], default="4h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # expand env like ${VAR}
    def expand_env(o):
        if isinstance(o, dict): return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list): return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"): return os.environ.get(o[2:-1], o)
        return o
    cfg = expand_env(cfg)

    backtest(cfg, universe=args.universe, tf=args.tf, start=args.start, end=args.end)

if __name__ == "__main__":
    main()
