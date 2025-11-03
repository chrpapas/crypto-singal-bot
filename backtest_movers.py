#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mover-focused trend backtester for MEXC.

Key ideas:
- Universe: liquid USDT spot coins, optionally filtered to "movers" (24h % + volume).
- Signals: day / swing / trend (same logic family as your bot).
- ETH gate: only go long when ETH is in healthy trend (EMA stack + RSI).
- Exits (standard mover exit):
    * 50% at +tp1_R R (default 1.5R),
    * Move stop on remainder to breakeven,
    * Trail remainder with ATR(atr_len) * atr_k (default 20 * 3.5),
    * Optional time stop after N bars (default 30).
- Intrabar assumption: stops trigger before targets on the same bar (conservative).
"""

import argparse, os, yaml, time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import ccxt

# ------------------------ small print helper ------------------------
def p(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

# ------------------------ TA helpers ------------------------
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

def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style ATR via EWM."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ------------------------ Signal defs (same logic family as your bot) ------------------------
def day_signal(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    look = int(params.get("lookback_high", 30))
    voln = int(params.get("vol_sma", 30))
    if len(df) < max(look, voln) + 5:
        return None

    volS = sma(df["volume"], voln)
    r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]

    highlvl = df["high"].iloc[-(look+1):-1].max()

    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = (
        breakout_edge
        and (last["volume"] > (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"]))
        and (params.get("rsi_min", 52) <= r.iloc[-1] <= params.get("rsi_max", 78))
    )

    retest_edge = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok   = (
        retest_edge
        and (last["volume"] > 0.8 * (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last["volume"]))
        and (r.iloc[-1] >= params.get("rsi_min", 52))
    )

    if not (breakout_ok or retrec_ok):
        return None

    entry = float(last["close"])
    stop  = float(min(df["low"].iloc[-10:]))

    return {
        "type": "day",
        "entry": entry,
        "stop": stop,
        "level": float(highlvl),
        "note": "Breakout" if breakout_ok else "Retest-Reclaim",
    }

def swing_signal(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    need = max(
        params.get('ema100',100),
        params.get('vol_sma',20),
        params.get('breakout_lookback',34)
    ) + 5
    if len(df) < need:
        return None

    df = df.copy()
    df['ema20']  = ema(df['close'], params.get('ema20',20))
    df['ema50']  = ema(df['close'], params.get('ema50',50))
    df['ema100'] = ema(df['close'], params.get('ema100',100))
    df['volS']   = sma(df['volume'], params.get('vol_sma',20))
    r            = rsi(df['close'],14)
    last         = df.iloc[-1]

    aligned = (last['ema20'] > last['ema50'] > last['ema100']) and (r.iloc[-1] >= params.get('rsi_min',50))
    within  = abs((last['close'] - last['ema20']) / max(1e-12,last['ema20']) * 100) <= params.get('pullback_pct_max',10.0)
    bounce  = last['close'] > df['close'].iloc[-2]

    hl      = df['high'].iloc[-(params.get('breakout_lookback',34)+1):-1].max()
    breakout = (
        (last['close'] > hl)
        and (last['volume'] > (df['volS'].iloc[-1] if not np.isnan(df['volS'].iloc[-1]) else last['volume']))
    )

    if not (aligned and ((within and bounce) or breakout)):
        return None

    entry = float(last['close'])
    stop  = float(min(df['low'].iloc[-10:]))

    return {
        "type": "swing",
        "entry": entry,
        "stop": stop,
        "level": float(hl),
        "note": "4h Pullback-Bounce" if (within and bounce) else "4h Breakout",
    }

def trend_signal(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    need = max(
        params.get("ema100",100),
        params.get("vol_sma",20),
        params.get("breakout_lookback",55)
    ) + 5
    if len(df) < need:
        return None

    df = df.copy()
    df["ema20"]  = ema(df["close"], params.get("ema20",20))
    df["ema50"]  = ema(df["close"], params.get("ema50",50))
    df["ema100"] = ema(df["close"], params.get("ema100",100))
    df["volS"]   = sma(df["volume"], params.get("vol_sma",20))
    r            = rsi(df["close"],14)
    last         = df.iloc[-1]

    aligned = (last["ema20"] > last["ema50"] > last["ema100"]) and (r.iloc[-1] >= params.get("rsi_min",50))
    within  = abs((last["close"]-last["ema20"])/max(1e-12,last["ema20"])*100) <= params.get("pullback_pct_max",10.0)
    bounce  = last["close"] > df["close"].iloc[-2]

    hl       = df["high"].iloc[-(params.get("breakout_lookback",55)+1):-1].max()
    breakout = (
        (last["close"] > hl)
        and (last["volume"] > (df["volS"].iloc[-1] if not np.isnan(df["volS"].iloc[-1]) else last["volume"]))
    )

    if not (aligned and ((within and bounce) or breakout)):
        return None

    entry = float(last["close"])
    stop  = float(min(df["low"].iloc[-10:]))

    return {
        "type": "trend",
        "entry": entry,
        "stop": stop,
        "level": float(hl),
        "note": "Pullback-Bounce" if (within and bounce) else "Breakout",
    }

# ------------------------ ETH gate ------------------------
def eth_gate_ok(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """
    gate_ok = (ema20 > ema50 if required) and (RSI >= min_rsi if set)
    """
    if df is None or df.empty:
        return pd.Series(False, index=pd.Index([], dtype='datetime64[ns, UTC]'))

    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    r   = rsi(df["close"],14)

    ok = pd.Series(True, index=df.index)
    if cfg.get("require_ema_stack", True):
        ok &= (e20 > e50)
    if cfg.get("min_rsi", None) is not None:
        ok &= (r >= float(cfg["min_rsi"]))
    return ok.rename("gate_ok")

# ------------------------ ccxt helpers ------------------------
def ccxt_timeframe_ms(tf: str) -> int:
    m = {
        "1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000,"30m":1_800_000,
        "1h":3_600_000,"2h":7_200_000,"4h":14_400_000,"1d":86_400_000
    }
    return m[tf]

def fetch_ohlcv_tz_aware(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Pull OHLCV iteratively between [start, end], return UTC tz-aware DataFrame.
    """
    tf_ms = ccxt_timeframe_ms(timeframe)
    since = int(start.timestamp()*1000)
    out   = []

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

def build_movers_timeline_5m(
    df5_by_pair: Dict[str, pd.DataFrame],
    min_change_24h_pct: float,
    min_vol_usd_24h: float,
    limit: int
) -> Dict[pd.Timestamp, List[str]]:
    all_ts = sorted(set().union(*[df.index for df in df5_by_pair.values()])) if df5_by_pair else []
    timeline: Dict[pd.Timestamp, List[str]] = {}

    for ts in all_ts:
        picks = []
        for pair, df in df5_by_pair.items():
            if ts not in df.index:
                continue
            row    = df.loc[ts]
            ret    = float(row.get("ret_24h") or 0.0)
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
DEFAULT_STABLES = {
    "USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT",
    "USD1","USDE","USDY","USDP","SUSD","EURS","EURT","PYUSD"
}

def is_stable_base(base: str, extra: List[str]) -> bool:
    b = base.upper().replace("3L","").replace("3S","").replace("5L","").replace("5S","")
    return (b in DEFAULT_STABLES) or (b in {e.upper() for e in (extra or [])})

def has_pair(ex: ccxt.Exchange, pair: str) -> bool:
    mkts = getattr(ex, "markets", None)
    if mkts is None:
        try:
            ex.load_markets()
        except Exception:
            return False
        mkts = ex.markets
    return pair in mkts or pair in getattr(ex, "symbols", [])

def mexc_volume_universe(
    ex: ccxt.Exchange,
    *,
    quote: str = "USDT",
    max_pairs: int = 80,
    min_usd_vol: float = 2_000_000,
    extra_stables=None
) -> List[str]:
    extra_stables = extra_stables or []
    try:
        p("[universe] candidates -> scanning /USDT spot tickers…")
        tickers = ex.fetch_tickers()
    except Exception as e:
        p("[universe] fetch_tickers err:", e)
        return []

    rows=[]
    for sym,t in tickers.items():
        if f"/{quote}" not in sym:
            continue
        base,_ = sym.split("/")
        if is_stable_base(base, extra_stables):
            continue
        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume") or 0
            last   = t.get("last") or t.get("close") or 0
            qv     = base_v * last
        try:
            qv = float(qv or 0)
        except:
            qv = 0.0
        if qv >= float(min_usd_vol):
            rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s,_ in rows[:max_pairs]]
    p(f"[universe] volume universe -> {len(pairs)} /{quote} spot pairs (ex-stables), min_vol=${min_usd_vol:,.0f}")
    return pairs

def cmc_top100_symbols(cfg: Dict[str,Any]) -> List[str]:
    key = os.environ.get("CMC_API_KEY") or cfg.get("movers",{}).get("cmc_api_key","")
    if not key:
        return []
    import requests
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    try:
        r = requests.get(
            url,
            headers={"X-CMC_PRO_API_KEY": key},
            params={"limit":140,"convert":"USD"},
            timeout=15
        )
        data = r.json().get("data", [])
        data.sort(key=lambda x: x.get("quote",{}).get("USD",{}).get("market_cap",0), reverse=True)
        return [it["symbol"].upper() for it in data[:110] if "symbol" in it]
    except Exception as e:
        p("[cmc] error:", e)
        return []

def map_to_mexc_pairs(
    ex: ccxt.Exchange,
    syms: List[str],
    quote: str = "USDT",
    extra_stables=None
) -> List[str]:
    extra_stables = extra_stables or []
    out=[]
    for s in syms:
        pair = f"{s}/{quote}"
        if not has_pair(ex, pair):
            continue
        base,_ = pair.split("/")
        if is_stable_base(base, extra_stables):
            continue
        out.append(pair)
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
    exit_cfg: Dict[str, Any],
    *,
    atr_col: str = "ATR",
) -> Tuple[float, str, pd.Timestamp]:
    """
    Standard "mover" exit:
      - Risk 1R = entry - stop.
      - 50% off at +tp1_R * R (default 1.5R).
      - Move stop to BE on remainder.
      - Trail remainder with chandelier: highest_close_since_TP1 - atr_k * ATR(atr_len).
      - Optional time stop after time_stop_bars bars (default 30).
    Intrabar priority:
      - Before TP1: SL first, then targets.
      - After TP1: BE, then trail, then TP2.
    """
    tp1_R = float(exit_cfg.get("tp1_R", 1.5))
    tp2_R = float(exit_cfg.get("tp2_R", 4.0))
    atr_len = int(exit_cfg.get("atr_len", 20))
    atr_k = float(exit_cfg.get("atr_k", 3.5))
    time_stop_bars = int(exit_cfg.get("time_stop_bars", 30))

    fwd = df.loc[df.index > ts_entry]
    if fwd.empty:
        return (0.0, "NO_DATA", ts_entry)

    R_unit = (entry - stop)
    if R_unit <= 0:
        return (0.0, "BAD_STOP", ts_entry)

    tp1_px = entry + tp1_R * R_unit
    tp2_px = entry + tp2_R * R_unit if tp2_R > 0 else None

    realized_R = 0.0
    have_tp1 = False
    bars_after_entry = 0
    highest_close_since_tp1 = -float("inf")
    last_ts_exit = ts_entry

    for ts, row in fwd.iterrows():
        bars_after_entry += 1
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        atr_now = float(row.get(atr_col, np.nan))

        # --- BEFORE TP1: full position still on ---
        if not have_tp1:
            # conservative: SL first
            if lo <= stop:
                return (-1.0, "SL", ts)

            hit_tp1 = hi >= tp1_px
            hit_tp2 = (tp2_px is not None) and (hi >= tp2_px)

            if hit_tp1:
                # realize 50% at tp1_R
                realized_R += 0.5 * tp1_R
                have_tp1 = True
                last_ts_exit = ts

                # same-bar BE check
                if lo <= entry:
                    return (realized_R, "TP1_then_BE_samebar", ts)

                # same-bar TP2 check
                if hit_tp2:
                    realized_R += 0.5 * tp2_R
                    return (realized_R, "TP2_after_TP1_samebar", ts)

                highest_close_since_tp1 = max(highest_close_since_tp1, cl)
            else:
                # time stop before TP1
                if time_stop_bars and bars_after_entry >= time_stop_bars:
                    R_now = (cl - entry) / R_unit
                    return (R_now, "TIME_STOP_BEFORE_TP1", ts)
                continue

        # --- AFTER TP1: remainder half with BE + chandelier trail ---
        highest_close_since_tp1 = max(highest_close_since_tp1, cl)
        trail = None
        if np.isfinite(atr_now):
            trail = highest_close_since_tp1 - atr_k * atr_now

        # BE first
        if lo <= entry:
            return (realized_R, "BE_after_TP1", ts)

        # trail next
        if trail is not None and lo <= trail:
            return (realized_R, f"TRAIL_hit_{atr_len}x{atr_k}", ts)

        # fixed TP2 for remainder
        if (tp2_px is not None) and (hi >= tp2_px):
            realized_R += 0.5 * tp2_R
            return (realized_R, "TP2_after_TP1", ts)

        # time stop after TP1
        if time_stop_bars and bars_after_entry >= time_stop_bars:
            rem_R = 0.5 * ((cl - entry) / R_unit)
            realized_R += rem_R
            return (realized_R, "TIME_STOP_AFTER_TP1", ts)

        last_ts_exit = ts

    # --- End of data: close at last close ---
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

    p(">>> backtest starting (mover-focused trend bot) <<<")

    # --- Universe selection ---
    extra_stables = cfg.get("filters",{}).get("extra_stables", [])
    pairs: List[str] = []

    if universe == "top100":
        syms = cmc_top100_symbols(cfg)
        if syms:
            pairs = map_to_mexc_pairs(ex, syms, "USDT", extra_stables)
            p(f"[universe] mapped top100 -> {len(pairs)} pairs")
        if not pairs:
            pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=80,
                                         min_usd_vol=2_000_000, extra_stables=extra_stables)

    elif universe == "volume":
        pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=80,
                                     min_usd_vol=2_000_000, extra_stables=extra_stables)

    elif universe == "movers":
        # candidate pool for movers comes from volume universe (slightly larger)
        pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=120,
                                     min_usd_vol=2_000_000, extra_stables=extra_stables)
    else:
        p(f"[universe] unknown '{universe}', defaulting to volume")
        pairs = mexc_volume_universe(ex, quote="USDT", max_pairs=80,
                                     min_usd_vol=2_000_000, extra_stables=extra_stables)

    p(f"[universe] {len(pairs)} pairs ({universe}) | tf={tf}")

    # --- ETH gate ---
    gate_cfg = cfg.get(
        "eth_gate",
        {"enabled": True, "require_ema_stack": True, "min_rsi": 52, "timeframe": tf}
    )
    gate_on = bool(gate_cfg.get("enabled", True))
    eth_gate_series = pd.DataFrame()
    eth_gate_keys: List[pd.Timestamp] = []

    if gate_on:
        try:
            gate_tf = str(gate_cfg.get("timeframe", tf))
            p(f"[eth-gate] fetching ETH/USDT {gate_tf} for gate...")
            df_eth = fetch_ohlcv_tz_aware(ex, "ETH/USDT", gate_tf,
                                          start_utc - pd.Timedelta(days=30), end_utc)
            if df_eth.empty:
                p("[eth-gate] series empty; gate will be OFF.")
                gate_on = False
            else:
                gate_ok_series = eth_gate_ok(df_eth, gate_cfg)
                eth_gate_series = pd.DataFrame({
                    "close": df_eth["close"],
                    "gate_ok": gate_ok_series
                })
                eth_gate_keys = list(eth_gate_series.index)
                last = eth_gate_series.iloc[-1]
                p(f"[eth-gate] {gate_tf}:ON -> gate_ok(last)={bool(last['gate_ok'])}")
        except Exception as e:
            p("[eth-gate] failed to fetch ETH/USDT; gate will be OFF.", e)
            gate_on = False
    else:
        p(f"[eth-gate] {tf}:OFF (config disabled)")

    # --- Main TF data + ATR ---
    tf_df: Dict[str, pd.DataFrame] = {}
    warmup_days = 90
    exit_cfg = cfg.get("exit_params", {
        "tp1_R": 1.5,
        "tp2_R": 4.0,
        "atr_len": 20,
        "atr_k": 3.5,
        "time_stop_bars": 30,
    })
    atr_len = int(exit_cfg.get("atr_len", 20))

    p(f"[data] fetching OHLCV for {len(pairs)} pairs, tf={tf}, with ~{warmup_days}d warmup…")
    for i, p_sym in enumerate(pairs, 1):
        try:
            df = fetch_ohlcv_tz_aware(
                ex, p_sym, tf,
                start_utc - pd.Timedelta(days=warmup_days),
                end_utc
            )
            if df is not None and not df.empty:
                df = df.copy()
                df["ATR"] = atr(df, n=atr_len)
                tf_df[p_sym] = df
        except Exception as e:
            p(f"[data] error fetching {p_sym}: {e}")
        if i % 20 == 0:
            p(f"  pulled {i}/{len(pairs)}…")

    if not tf_df:
        p("No data fetched on main TF; aborting.")
        return

    p(f"[universe] {len(tf_df)} pairs with sufficient history | tf={tf}")

    # --- Movers timeline from 5m ---
    mv_cfg = cfg.get("movers", {
        "min_change_24h": 25.0,
        "min_volume_usd_24h": 10_000_000.0,
        "limit": 40
    })
    df5_by_pair: Dict[str, pd.DataFrame] = {}
    if universe == "movers":
        p("[movers] building 5m movers timeline…")
        for i, p_sym in enumerate(pairs, 1):
            try:
                df5 = fetch_ohlcv_tz_aware(
                    ex, p_sym, "5m",
                    start_utc - pd.Timedelta(days=3),
                    end_utc
                )
                if df5 is None or df5.empty:
                    continue
                df5 = add_rolling_5m_metrics(df5)
                df5_by_pair[p_sym] = df5
            except Exception as e:
                p(f"[movers] error fetching 5m for {p_sym}: {e}")
            if i % 20 == 0:
                p(f"  movers 5m fetched for {i}/{len(pairs)}…")

        movers_timeline = build_movers_timeline_5m(
            df5_by_pair,
            float(mv_cfg.get("min_change_24h", 25.0)),
            float(mv_cfg.get("min_volume_usd_24h", 10_000_000.0)),
            int(mv_cfg.get("limit", 40))
        )
        timeline_keys = sorted(movers_timeline.keys())
        p(f"[movers] timeline built with {len(timeline_keys)} snapshots")
    else:
        movers_timeline, timeline_keys = {}, []

    # --- Strategy parameters ---
    dayP   = cfg.get("day_trade_params", {
        "lookback_high":30,
        "vol_sma":30,
        "rsi_min":52,
        "rsi_max":78
    })
    swingP = cfg.get("swing_trade_params", {
        "ema20":20,
        "ema50":50,
        "ema100":100,
        "rsi_min":55,
        "pullback_pct_max":8.0,
        "vol_sma":20,
        "breakout_lookback":34
    })
    trendP = cfg.get("trend_trade_params", {
        "ema20":20,
        "ema50":50,
        "ema100":100,
        "rsi_min":55,
        "pullback_pct_max":8.0,
        "vol_sma":20,
        "breakout_lookback":55
    })

    # --- Backtest loop: walk bar-by-bar, evaluate signals ---
    hits: List[Tuple[pd.Timestamp,str,str,str,float,float]] = []

    common_ts = sorted(set().union(*[df.index for df in tf_df.values()]))
    common_ts = [t for t in common_ts if (t >= start_utc and t <= end_utc)]
    if not common_ts:
        p("No bars in selected window after slicing.")
        return

    p(f"[loop] bars in window: {len(common_ts)}")

    for ts in common_ts:
        # pick active universe
        if universe == "movers":
            snap = last_snapshot_leq(timeline_keys, ts)
            scan_pairs = movers_timeline.get(snap, []) if snap else []
        else:
            scan_pairs = list(tf_df.keys())
        if not scan_pairs:
            continue

        # ETH gate at this time (using latest <= ts from gate TF)
        gate_block = False
        if gate_on and eth_gate_keys:
            snap_gate = last_snapshot_leq(eth_gate_keys, ts)
            if snap_gate is not None:
                gate_block = not bool(eth_gate_series.loc[snap_gate, "gate_ok"])

        for pair in scan_pairs:
            dfp = tf_df.get(pair)
            if dfp is None or (ts not in dfp.index):
                continue
            dfx = dfp.loc[:ts]
            if len(dfx) < 60:
                continue

            sig = None
            for fn, par, name in (
                (day_signal,   dayP,   "day"),
                (swing_signal, swingP, "swing"),
                (trend_signal, trendP, "trend"),
            ):
                s = fn(dfx, par)
                if s:
                    s["type"] = name
                    sig = s
                    break

            if not sig:
                continue
            if gate_block:
                # ETH gate blocking longs in bad regime
                continue

            hits.append((
                ts, pair, tf, sig["type"],
                float(sig["entry"]), float(sig["stop"])
            ))

    p(f"[loop] collected signals: {len(hits)}")

    # --- Report & Performance ---
    p("\n=== Backtest Summary ===")
    p(f"Bars tested: {len(common_ts)}  | Pairs: {len(tf_df)}  | Universe: {universe.upper()}  | TF: {tf}")
    p(f"Signals generated: {len(hits)}")

    if hits:
        p("\nFirst 20 signals:")
        for ts, pair, tfv, typ, entry, stop in hits[:20]:
            p(f"  {ts.isoformat()}  {pair} {tfv} {typ}  entry={entry:.6f} stop={stop:.6f}")

        # simulate trades
        rows = []
        for ts, pair, tfv, typ, entry, stop in hits:
            dfp = tf_df.get(pair)
            if dfp is None:
                continue
            R, reason, ts_exit = simulate_trade_standard_exit(
                dfp, ts, entry, stop, exit_cfg, atr_col="ATR"
            )
            R_unit = (entry - stop)
            tp1_R = float(exit_cfg.get("tp1_R", 1.5))
            tp2_R = float(exit_cfg.get("tp2_R", 4.0))
            t1_used = entry + tp1_R * R_unit
            t2_used = entry + tp2_R * R_unit if tp2_R > 0 else None
            rows.append({
                "ts_entry": ts,
                "pair":     pair,
                "tf":       tfv,
                "type":     typ,
                "entry":    entry,
                "stop":     stop,
                "t1_used":  t1_used,
                "t2_used":  t2_used,
                "R":        R,
                "exit_reason": reason,
                "ts_exit":  ts_exit,
            })

        if not rows:
            p("\n[perf] No trades could be simulated (no forward bars).")
            return

        perf = pd.DataFrame(rows).sort_values("ts_entry").reset_index(drop=True)
        perf["cum_R"] = perf["R"].cumsum()

        # stats
        n       = len(perf)
        wins    = int((perf["R"] > 0).sum())
        losses  = n - wins
        win_rate = (wins / n * 100.0) if n else 0.0
        avg_R    = float(perf["R"].mean())
        exp_R    = avg_R
        gross_win  = float(perf.loc[perf["R"] > 0, "R"].sum())
        gross_loss = -float(perf.loc[perf["R"] < 0, "R"].sum())
        profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else float("inf")
        mdd_R = max_drawdown(perf["cum_R"])

        tp1_R = float(exit_cfg.get("tp1_R", 1.5))
        tp2_R = float(exit_cfg.get("tp2_R", 4.0))
        atr_k = float(exit_cfg.get("atr_k", 3.5))

        p(f"\n=== Performance (R multiples, exit: 50% @{tp1_R}R, BE, ATR({atr_len})*{atr_k} trail, TP2 {tp2_R}R) ===")
        p(f"Trades: {n} | Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.1f}%")
        p(f"Avg R/trade: {avg_R:.3f} | Expectancy: {exp_R:.3f} R")
        p(f"Profit Factor: {profit_factor:.2f}")
        p(f"Total R: {float(perf['R'].sum()):.2f} | Max Drawdown: {mdd_R:.2f} R")

        # save
        out_dir = "./backtest_out"
        os.makedirs(out_dir, exist_ok=True)
        perf.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        perf[["ts_entry","cum_R"]].to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)
        p(f"[saved] {out_dir}/trades.csv  and  {out_dir}/equity_curve.csv")

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_trader_bot_config.yml")
    ap.add_argument("--universe", choices=["top100","movers","volume"], default="movers")
    ap.add_argument("--tf", choices=["1h","4h","1d"], default="4h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # expand ${VAR} from environment
    def expand_env(o):
        if isinstance(o, dict):
            return {k: expand_env(v) for k,v in o.items()}
        if isinstance(o, list):
            return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o

    cfg = expand_env(cfg)

    backtest(cfg, universe=args.universe, tf=args.tf, start=args.start, end=args.end)

if __name__ == "__main__":
    main()
