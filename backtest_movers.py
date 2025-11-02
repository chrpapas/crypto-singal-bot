#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtester for MEXC signal logic with ETH-gate and historical Movers timeline (built from 5m MEXC data).

Run examples:
  python3 backtest_mexc_bot.py --config mexc_trader_bot_config.yml --universe top100 --tf 4h --start 2025-07-01 --end 2025-10-29
  python3 backtest_mexc_bot.py --config mexc_trader_bot_config.yml --universe movers --tf 4h --start 2025-07-01 --end 2025-10-29

Notes
- Uses ccxt.mexc for OHLCV (spot).
- Movers are reconstructed historically from 5m bars (rolling 24h % and 24h USD volume).
- ETH gate is controlled by config (eth_gate.enabled). If disabled in YAML, it’s OFF even if CLI “gate_on” default is true.
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
        # advance by one tf to avoid duplicates
        next_since = last_ts + tf_ms
        if next_since >= int(end.timestamp()*1000):
            break
        # safety stop if not advancing
        if next_since <= since:
            break
        since = next_since
        # be gentle
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
    # union of 5m timestamps
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

    # Fetch main TF data for all pairs
    tf_df: Dict[str, pd.DataFrame] = {}
    warmup_days = 90
    for p in pairs:
        try:
            df = fetch_ohlcv_tz_aware(ex, p, tf, start_utc - pd.Timedelta(days=warmup_days), end_utc)
            if df is not None and not df.empty:
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
    dayP = cfg.get("day_trade_params", {"lookback_high":30, "vol_sma":30, "rsi_min":52, "rsi_max":78})
    swingP = cfg.get("swing_trade_params", {"ema20":20,"ema50":50,"ema100":100,"rsi_min":50,"pullback_pct_max":10.0,"vol_sma":20,"breakout_lookback":34})
    trendP = cfg.get("trend_trade_params", {"ema20":20,"ema50":50,"ema100":100,"rsi_min":50,"pullback_pct_max":10.0,"vol_sma":20,"breakout_lookback":55})

    # Backtest loop: walk over TF bars; for movers, choose universe from last 5m snapshot ≤ bar time
    total = 0
    hits: List[Tuple[pd.Timestamp,str,str,str,float,float]] = []  # (ts, pair, tf, typ, entry, stop)

    # common TF timestamps (intersect across pairs to be safe)
    common_ts = sorted(set().union(*[df.index for df in tf_df.values()]))
    # cut to [start,end]
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

        # If ETH gate enabled and bearish at ts -> skip bullish entries
        gate_block = False
        if gate_on and (ts in eth_gate_series.index):
            gate_block = not bool(eth_gate_series.loc[ts, "gate_ok"])

        for pair in scan_pairs:
            df = tf_df.get(pair)
            if df is None or (ts not in df.index):
                continue
            # slice up to ts to avoid look-ahead
            dfx = df.loc[:ts]
            if len(dfx) < 60:
                continue

            # Try day, swing, trend
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
                # ETH gate blocks bullish entries
                continue

            total += 1
            hits.append((ts, pair, tf, sig["type"], sig["entry"], sig["stop"]))

    # Report
    print(f"\n=== Backtest Summary ===")
    print(f"Bars tested: {len(common_ts)}  | Pairs: {len(pairs)}  | Universe: {universe.upper()}  | TF: {tf}")
    print(f"Signals generated: {len(hits)}")
    if hits:
        # print first 20 for sanity
        print("\nFirst 20 signals:")
        for ts, pair, tfv, typ, entry, stop in hits[:20]:
            print(f"  {ts.isoformat()}  {pair} {tfv} {typ}  entry={entry:.6f} stop={stop:.6f}")

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
