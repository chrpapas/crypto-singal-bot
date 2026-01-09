import os
import time
import json
import math
import hashlib
from typing import Any, Dict, Optional, List, Tuple

import yaml
import requests
import numpy as np
import pandas as pd
import ccxt
from redis import Redis


# =========================
# Config helpers
# =========================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def now_ts() -> int:
    return int(time.time())

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def timeframe_to_seconds(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")

def is_candle_closed(candle_open_ts: int, tf: str, now: Optional[int] = None) -> bool:
    if now is None:
        now = now_ts()
    return (candle_open_ts + timeframe_to_seconds(tf)) <= now

def round_price(x: float) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    digits = max(0, 6 - int(math.floor(math.log10(abs(x)))) - 1)
    return round(x, digits)

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# =========================
# Redis keys
# =========================

def strategy_key(cfg: dict) -> str:
    s = cfg["strategy"]
    return f"{s['name']}:{s.get('version','v0')}"

def k_lock(lock_key: str) -> str:
    return lock_key

def k_last_candle(market: str, symbol: str, tf: str) -> str:
    return f"last_candle:{market}:{symbol}:{tf}"

def k_latest(exchange: str, market: str, symbol: str, tf: str) -> str:
    return f"signal:latest:{exchange}:{market}:{symbol}:{tf}"

def k_history(exchange: str, market: str, symbol: str, tf: str) -> str:
    return f"signal:history:{exchange}:{market}:{symbol}:{tf}"

def k_published(signal_id: str) -> str:
    return f"published:{signal_id}"

def k_universe(exchange: str, market: str) -> str:
    return f"universe:{exchange}:{market}"

def k_universe_ts(exchange: str, market: str) -> str:
    return f"universe_ts:{exchange}:{market}"

# Performance keys (versioned)
def k_open_set(sk: str) -> str:
    return f"trade:open:set:{sk}"

def k_open_trade(sk: str, signal_id: str) -> str:
    return f"trade:open:{sk}:{signal_id}"

def k_closed_list(sk: str) -> str:
    return f"trade:closed:{sk}"  # append-only list (LPUSH)

def k_metrics(sk: str) -> str:
    return f"metrics:{sk}"


# =========================
# Redis primitives
# =========================

def acquire_lock(r: Redis, key: str, ttl: int) -> bool:
    return bool(r.set(key, "1", nx=True, ex=ttl))

def already_processed_candle(r: Redis, market: str, symbol: str, tf: str, candle_ts: int) -> bool:
    key = k_last_candle(market, symbol, tf)
    last = r.get(key)
    if last is not None and int(last) >= candle_ts:
        return True
    r.set(key, str(candle_ts), ex=14 * 24 * 3600)
    return False

def write_signal_to_redis(r: Redis, sig: Dict[str, Any]) -> None:
    key_latest = k_latest(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"])
    key_hist = k_history(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"])
    r.set(key_latest, json.dumps(sig), ex=6 * 3600)
    r.lpush(key_hist, json.dumps(sig))
    r.ltrim(key_hist, 0, 200)

def mark_published(r: Redis, signal_id: str, ttl_days: int) -> bool:
    return bool(r.set(k_published(signal_id), "1", nx=True, ex=ttl_days * 86400))

def cooldown_key(exchange: str, market: str, symbol: str, tf: str, setup: str, sk: str) -> str:
    return f"cooldown:{sk}:{exchange}:{market}:{symbol}:{tf}:{setup}"

def set_cooldown(r: Redis, key: str, seconds: int) -> None:
    r.set(key, "1", ex=seconds)

def in_cooldown(r: Redis, key: str) -> bool:
    return r.get(key) is not None


# =========================
# Indicators
# =========================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# =========================
# Discord
# =========================

def format_discord_message(sig: Dict[str, Any]) -> str:
    side = sig["side"]
    sym = sig["symbol"]
    tf = sig["timeframe"]
    conf = sig["confidence"]
    entry = sig["entry"]["price"]
    stop = sig["exit"]["stop"]
    tp1, tp2 = sig["exit"]["take_profit"]

    risk = abs(entry - stop)
    r1 = abs(tp1 - entry) / risk if risk > 0 else 0
    r2 = abs(tp2 - entry) / risk if risk > 0 else 0

    setup = sig.get("setup", "signal")
    ver = sig.get("strategy_version", "")

    return (
        f"**{setup}** `{ver}` | **{side}** `{sym}` • **{tf}** • conf: **{conf}**\n"
        f"Entry: `{entry}` | Stop: `{stop}`\n"
        f"TP1: `{tp1}` (≈ {r1:.2f}R) | TP2: `{tp2}` (≈ {r2:.2f}R)\n"
        f"Trail: `{sig['exit']['trail']['method']}` x `{sig['exit']['trail']['mult']}` "
        f"(activation: `{sig['exit']['trail']['activation']}`)\n"
        f"Invalidation: _{sig['invalidation']}_\n"
        f"Signal ID: `{sig['signal_id']}`"
    )

def post_to_discord(webhook_url: str, content: str, username: str = "", avatar_url: str = "") -> None:
    payload = {"content": content}
    if username:
        payload["username"] = username
    if avatar_url:
        payload["avatar_url"] = avatar_url
    resp = requests.post(webhook_url, json=payload, timeout=15)
    resp.raise_for_status()


# =========================
# CCXT: MEXC perps
# =========================

def build_mexc_client(cfg: dict) -> ccxt.Exchange:
    mexc_cfg = cfg.get("mexc", {})
    key_env = mexc_cfg.get("api_key_env", "MEXC_KEY")
    sec_env = mexc_cfg.get("api_secret_env", "MEXC_SECRET")

    api_key = os.environ.get(key_env, "")
    api_secret = os.environ.get(sec_env, "")

    return ccxt.mexc({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

def normalize_symbol_for_ccxt(symbol: str) -> str:
    s = symbol.strip()
    if ":" in s and "/" in s:
        return s
    if "_" in s:
        base, quote = s.split("_", 1)
        quote_u = quote.upper()
        return f"{base}/{quote_u}:{quote_u}"
    if "/" in s:
        base, quote = s.split("/", 1)
        quote_u = quote.upper()
        return f"{base}/{quote_u}:{quote_u}"
    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        return f"{base}/USDT:USDT"
    return s

def fetch_ohlcv_ccxt(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ccxt_symbol = normalize_symbol_for_ccxt(symbol)
    ohlcv = ex.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise RuntimeError(f"No OHLCV returned for {ccxt_symbol} {timeframe}")

    out = []
    for row in ohlcv:
        ms = int(row[0])
        ts = ms // 1000
        out.append({
            "ts": ts,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]) if len(row) > 5 else 0.0,
        })
    return pd.DataFrame(out).dropna().sort_values("ts").reset_index(drop=True)

def latest_closed_candle_ts(df: pd.DataFrame, tf: str) -> int:
    if len(df) < 2:
        return int(df.iloc[-1]["ts"])
    last_ts = int(df.iloc[-1]["ts"])
    return last_ts if is_candle_closed(last_ts, tf) else int(df.iloc[-2]["ts"])


# =========================
# Universe
# =========================

def build_universe_symbols(ex: ccxt.Exchange, cfg: dict) -> List[str]:
    ucfg = cfg.get("universe", {})
    quote = (ucfg.get("quote", "USDT") or "USDT").upper()
    require_colon_quote = bool(ucfg.get("require_colon_quote", True))
    top_n = int(ucfg.get("top_n", 200))
    min_qv = float(ucfg.get("min_quote_volume_usdt", 0))
    whitelist = set(ucfg.get("whitelist", []) or [])
    blacklist = set(ucfg.get("blacklist", []) or [])

    markets = ex.markets or {}
    candidates: List[str] = []

    for sym, m in markets.items():
        if not m or not m.get("active", True):
            continue
        if m.get("type") != "swap" and not m.get("swap", False):
            continue
        if (m.get("quote") or "").upper() != quote:
            continue
        if m.get("inverse", False) is True:
            continue
        if require_colon_quote and f":{quote}" not in sym:
            continue
        if whitelist and sym not in whitelist:
            continue
        if sym in blacklist:
            continue
        candidates.append(sym)

    if not candidates:
        return []

    try:
        tickers = ex.fetch_tickers(candidates)
    except Exception:
        tickers = ex.fetch_tickers()

    scored: List[Tuple[str, float]] = []
    for sym in candidates:
        t = tickers.get(sym) or {}
        qv = t.get("quoteVolume")
        if qv is None:
            bv = t.get("baseVolume")
            last = t.get("last")
            if bv is not None and last is not None:
                try:
                    qv = float(bv) * float(last)
                except Exception:
                    qv = None
        if qv is None:
            continue
        qv = float(qv)
        if qv < min_qv:
            continue
        scored.append((sym, qv))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]]

def get_universe_symbols_cached(r: Redis, ex: ccxt.Exchange, cfg: dict, exchange: str, market: str) -> List[str]:
    ucfg = cfg.get("universe", {})
    refresh = int(ucfg.get("refresh_seconds", 3600))

    ts_key = k_universe_ts(exchange, market)
    uni_key = k_universe(exchange, market)

    last_ts = r.get(ts_key)
    if last_ts is not None:
        age = now_ts() - int(last_ts)
        if age < refresh:
            raw = r.get(uni_key)
            if raw:
                try:
                    return json.loads(raw)
                except Exception:
                    pass

    syms = build_universe_symbols(ex, cfg)
    r.set(ts_key, str(now_ts()), ex=7 * 24 * 3600)
    r.set(uni_key, json.dumps(syms), ex=7 * 24 * 3600)
    return syms

def shard_symbols(r: Redis, symbols: List[str], cfg: dict) -> List[str]:
    ucfg = cfg.get("universe", {})
    shard_cfg = (ucfg.get("shard", {}) or {})
    if not bool(shard_cfg.get("enable", True)):
        return symbols

    per_run = int(shard_cfg.get("symbols_per_run", 60))
    step = int(shard_cfg.get("step", per_run))
    state_key = shard_cfg.get("state_key", "universe_shard_index")

    n = len(symbols)
    if n == 0:
        return []

    idx_raw = r.get(state_key)
    idx = int(idx_raw) if idx_raw is not None else 0
    idx %= n

    end = idx + per_run
    chunk = symbols[idx:end] if end <= n else symbols[idx:] + symbols[:(end % n)]

    next_idx = (idx + step) % n
    r.set(state_key, str(next_idx), ex=30 * 24 * 3600)
    return chunk


# =========================
# Strategy + confidence
# =========================

def candle_momentum_ratio(row: pd.Series) -> float:
    hi = float(row["high"])
    lo = float(row["low"])
    cl = float(row["close"])
    rng = hi - lo
    if not math.isfinite(rng) or rng <= 0:
        return 0.5
    return (cl - lo) / rng  # 0..1 (close near high = 1)

def compute_confidence_components(cfg: dict, df: pd.DataFrame, side: str) -> Dict[str, float]:
    s = cfg["strategy"]
    last = df.iloc[-1]

    atrv = float(last["atr"])
    if not math.isfinite(atrv) or atrv <= 0:
        return {"confidence": 0.0}

    close = float(last["close"])
    ema_tr = float(last["ema_trend"])
    slope = float(last["ema_trend_slope"])  # EMA trend slope over lookback
    slope_atr = slope / atrv  # signed
    dist_atr = abs((close - ema_tr) / atrv)  # magnitude

    # pullback depth: how far previous close was beyond ema_fast
    prev = df.iloc[-2]
    prev_close = float(prev["close"])
    prev_ema_fast = float(prev["ema_fast"])
    pull_depth_atr = abs(prev_ema_fast - prev_close) / atrv

    mom = candle_momentum_ratio(last)
    mom_for_side = mom if side == "LONG" else (1.0 - mom)  # short wants close near low

    # normalize to 0..1
    # trend strength: require sign; score grows from min -> (min+1)
    min_slope = float(s.get("trend_slope_min_atr", 0.2))
    ts = abs(slope_atr)
    trend_strength = clamp((ts - min_slope) / 1.0, 0.0, 1.0)

    # distance from EMA trend: prefer not too close, not too far (peak around ~1 ATR)
    dist_score = 1.0 - clamp(abs(dist_atr - 1.0) / 1.5, 0.0, 1.0)

    # pullback depth: prefer within bounds, peak around mid
    pb_min = float(s.get("pullback_min_atr", 0.15))
    pb_max = float(s.get("pullback_max_atr", 1.5))
    if pull_depth_atr < pb_min or pull_depth_atr > pb_max:
        pull_score = 0.0
    else:
        mid = (pb_min + pb_max) / 2.0
        pull_score = 1.0 - clamp(abs(pull_depth_atr - mid) / (pb_max - pb_min), 0.0, 1.0)

    # momentum: map mom_for_side from [0.5..1] -> [0..1]
    momentum_score = clamp((mom_for_side - 0.5) / 0.5, 0.0, 1.0)

    # combine
    confidence = clamp(0.35 * trend_strength + 0.25 * dist_score + 0.20 * pull_score + 0.20 * momentum_score, 0.0, 1.0)

    return {
        "confidence": float(confidence),
        "trend_strength": float(trend_strength),
        "dist_score": float(dist_score),
        "pull_score": float(pull_score),
        "momentum_score": float(momentum_score),
        "slope_atr": float(slope_atr),
        "dist_atr": float(dist_atr),
        "pull_depth_atr": float(pull_depth_atr),
        "mom": float(mom),
    }

def build_trend_pullback_signal(cfg: dict, df: pd.DataFrame, symbol: str, tf: str, exchange: str, market: str) -> Optional[Dict[str, Any]]:
    s = cfg["strategy"]
    ema_fast_n = int(s["ema_fast"])
    ema_trend_n = int(s["ema_trend"])
    atr_n = int(s["atr_period"])
    slope_lb = int(s.get("trend_slope_lookback", 10))

    if len(df) < max(ema_trend_n, atr_n) + slope_lb + 5:
        return None

    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast_n)
    df["ema_trend"] = ema(df["close"], ema_trend_n)
    df["atr"] = atr(df, atr_n)
    df["ema_trend_slope"] = df["ema_trend"] - df["ema_trend"].shift(slope_lb)
    df["atr_pct"] = df["atr"] / df["close"]

    last = df.iloc[-1]
    prev = df.iloc[-2]
    candle_ts = int(last["ts"])
    if not is_candle_closed(candle_ts, tf):
        return None

    close = float(last["close"])
    ema_fast_v = float(last["ema_fast"])
    ema_trend_v = float(last["ema_trend"])
    atr_v = float(last["atr"])

    if not (math.isfinite(atr_v) and atr_v > 0):
        return None

    # --- New: ATR% filter
    atr_pct = float(last["atr_pct"])
    atr_pct_min = float(s.get("atr_pct_min", 0.0))
    atr_pct_max = float(s.get("atr_pct_max", 999.0))
    if not (atr_pct_min <= atr_pct <= atr_pct_max):
        return None

    # --- New: trend slope filter (signed)
    slope = float(last["ema_trend_slope"])
    slope_atr = slope / atr_v
    min_slope_atr = float(s.get("trend_slope_min_atr", 0.2))

    trend_up = close > ema_trend_v and slope_atr >= min_slope_atr
    trend_down = close < ema_trend_v and slope_atr <= -min_slope_atr

    # --- Pullback reclaim condition
    pullback_long = trend_up and float(prev["close"]) < float(prev["ema_fast"]) and close > ema_fast_v
    pullback_short = trend_down and float(prev["close"]) > float(prev["ema_fast"]) and close < ema_fast_v
    if not (pullback_long or pullback_short):
        return None

    side = "LONG" if pullback_long else "SHORT"

    # --- New: pullback depth filter
    prev_close = float(prev["close"])
    prev_ema_fast = float(prev["ema_fast"])
    pull_depth_atr = abs(prev_ema_fast - prev_close) / atr_v
    pb_min = float(s.get("pullback_min_atr", 0.15))
    pb_max = float(s.get("pullback_max_atr", 1.5))
    if pull_depth_atr < pb_min or pull_depth_atr > pb_max:
        return None

    # --- New: candle momentum filter
    mom = candle_momentum_ratio(last)
    mom_for_side = mom if side == "LONG" else (1.0 - mom)
    mom_min = float(s.get("momentum_min", 0.55))
    if mom_for_side < mom_min:
        return None

    # Entry model
    entry = close

    # Stops/targets
    stop_mult = float(s["stop_atr_mult"])
    tp1_r = float(s["tp1_r"])
    tp2_r = float(s["tp2_r"])
    trail_mult = float(s["trail_atr_mult"])
    trail_activation = str(s.get("trail_activation", "after_tp1"))

    stop = entry - stop_mult * atr_v if side == "LONG" else entry + stop_mult * atr_v
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    tp1 = entry + tp1_r * risk if side == "LONG" else entry - tp1_r * risk
    tp2 = entry + tp2_r * risk if side == "LONG" else entry - tp2_r * risk

    comps = compute_confidence_components(cfg, df, side)
    confidence = float(comps["confidence"])

    raw_id = json.dumps({
        "exchange": exchange,
        "market": market,
        "symbol": symbol,
        "tf": tf,
        "candle_ts": candle_ts,
        "setup": s["name"],
        "version": s.get("version", "v0"),
        "side": side,
    }, sort_keys=True)
    signal_id = sha256_hex(raw_id)[:24]

    sig = {
        "signal_id": signal_id,
        "ts": candle_ts,
        "exchange": exchange,
        "market": market,
        "symbol": symbol,
        "timeframe": tf,
        "setup": s["name"],
        "strategy_version": s.get("version", "v0"),
        "side": side,
        "confidence": round(confidence, 4),
        "confidence_components": {k: round(float(v), 4) for k, v in comps.items() if k != "confidence"},
        "entry_model": "signal_close",
        "entry": {"type": "market", "price": round_price(entry)},
        "exit": {
            "stop": round_price(stop),
            "take_profit": [round_price(tp1), round_price(tp2)],
            "trail": {
                "method": "atr_chandelier",
                "mult": trail_mult,
                "activation": trail_activation,
            },
            "breakeven_after_tp1": bool(s.get("breakeven_after_tp1", True)),
            "breakeven_offset_r": float(s.get("breakeven_offset_r", 0.0)),
        },
        "invalidation": f"Trend weakens (EMA{int(s['ema_trend'])} slope flips) or close crosses EMA{int(s['ema_trend'])} against direction",
        "rationale": [
            f"Trend: close vs EMA{ema_trend_n} + slope filter (lookback {slope_lb})",
            f"Pullback: reclaim EMA{ema_fast_n}",
            f"ATR% filter: [{s.get('atr_pct_min')}..{s.get('atr_pct_max')}]",
            f"Momentum filter: >= {mom_min}",
        ],
        "perf": {
            "tp1_fraction": 0.5,
            "tp2_fraction": 0.5,
        }
    }
    return sig


# =========================
# Performance tracking (versioned)
# =========================

def get_metrics(r: Redis, sk: str) -> Dict[str, Any]:
    raw = r.get(k_metrics(sk))
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "sum_r": 0.0,
        "equity_r": 0.0,
        "peak_equity_r": 0.0,
        "max_drawdown_r": 0.0,
        "last_update_ts": 0,
    }

def save_metrics(r: Redis, sk: str, m: Dict[str, Any]) -> None:
    r.set(k_metrics(sk), json.dumps(m), ex=180 * 24 * 3600)

def update_metrics_on_close(r: Redis, sk: str, closed_trade: Dict[str, Any]) -> None:
    m = get_metrics(r, sk)
    m["trades"] += 1
    r_mult = float(closed_trade["r_multiple"])
    m["sum_r"] = float(m["sum_r"]) + r_mult
    if r_mult > 0:
        m["wins"] += 1
    else:
        m["losses"] += 1
    m["equity_r"] = float(m["equity_r"]) + r_mult
    m["peak_equity_r"] = max(float(m["peak_equity_r"]), float(m["equity_r"]))
    dd = float(m["peak_equity_r"]) - float(m["equity_r"])
    m["max_drawdown_r"] = max(float(m["max_drawdown_r"]), dd)
    m["last_update_ts"] = now_ts()
    save_metrics(r, sk, m)

def open_trade_exists(r: Redis, sk: str, signal_id: str) -> bool:
    return r.exists(k_open_trade(sk, signal_id)) == 1

def r_for_exit(side: str, entry: float, exit_price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    return (exit_price - entry) / risk if side == "LONG" else (entry - exit_price) / risk

def create_open_trade(r: Redis, sk: str, sig: Dict[str, Any], cfg: dict) -> None:
    sid = sig["signal_id"]
    if open_trade_exists(r, sk, sid):
        return

    s = cfg["strategy"]
    entry = float(sig["entry"]["price"])
    stop0 = float(sig["exit"]["stop"])
    tp1 = float(sig["exit"]["take_profit"][0])
    tp2 = float(sig["exit"]["take_profit"][1])
    side = sig["side"]
    risk = abs(entry - stop0)
    if risk <= 0:
        return

    trail_mult = float(sig["exit"]["trail"]["mult"])
    trail_activation = sig["exit"]["trail"].get("activation", "after_tp1")

    trade = {
        "signal_id": sid,
        "strategy_key": sk,
        "exchange": sig["exchange"],
        "market": sig["market"],
        "symbol": sig["symbol"],
        "timeframe": sig["timeframe"],
        "side": side,
        "setup": sig.get("setup", ""),
        "strategy_version": sig.get("strategy_version", "v0"),

        "opened_ts": int(sig["ts"]),
        "entry": entry,
        "stop0": stop0,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,

        "tp1_fraction": float(sig.get("perf", {}).get("tp1_fraction", 0.5)),
        "tp2_fraction": float(sig.get("perf", {}).get("tp2_fraction", 0.5)),

        "trail_mult": trail_mult,
        "trail_activation": trail_activation,
        "trail_require_trend_valid": bool(s.get("trail_require_trend_valid", True)),
        "ema_fast_n": int(s.get("ema_fast", 20)),
        "atr_period": int(s.get("atr_period", 14)),

        "breakeven_after_tp1": bool(sig["exit"].get("breakeven_after_tp1", True)),
        "breakeven_offset_r": float(sig["exit"].get("breakeven_offset_r", 0.0)),

        "tp1_hit": False,
        "current_stop": stop0,
        "highest_high": entry,
        "lowest_low": entry,
        "last_checked_ts": int(sig["ts"]),
        "_tp1_r_cache": None,
    }

    pipe = r.pipeline()
    pipe.set(k_open_trade(sk, sid), json.dumps(trade), ex=30 * 24 * 3600)
    pipe.sadd(k_open_set(sk), sid)
    pipe.execute()

def close_trade(r: Redis, sk: str, trade: Dict[str, Any], exit_price: float, exit_ts: int, reason: str, total_r: float, keep: int) -> None:
    closed = {
        "signal_id": trade["signal_id"],
        "strategy_key": sk,
        "exchange": trade["exchange"],
        "market": trade["market"],
        "symbol": trade["symbol"],
        "timeframe": trade["timeframe"],
        "side": trade["side"],
        "setup": trade.get("setup", ""),
        "strategy_version": trade.get("strategy_version", "v0"),
        "opened_ts": trade["opened_ts"],
        "closed_ts": exit_ts,
        "entry": trade["entry"],
        "stop0": trade["stop0"],
        "tp1": trade["tp1"],
        "tp2": trade["tp2"],
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "r_multiple": round(float(total_r), 6),
    }

    pipe = r.pipeline()
    pipe.lpush(k_closed_list(sk), json.dumps(closed))
    pipe.ltrim(k_closed_list(sk), 0, max(1000, int(keep)))
    pipe.delete(k_open_trade(sk, trade["signal_id"]))
    pipe.srem(k_open_set(sk), trade["signal_id"])
    pipe.execute()

    update_metrics_on_close(r, sk, closed)

def candle_hits(side: str, candle: Dict[str, float], level: float, kind: str) -> bool:
    if side == "LONG":
        return candle["low"] <= level if kind == "stop" else candle["high"] >= level
    else:
        return candle["high"] >= level if kind == "stop" else candle["low"] <= level

def update_trade_with_candles(trade: Dict[str, Any], df: pd.DataFrame) -> Optional[Tuple[float, int, str, float]]:
    side = trade["side"]
    entry = float(trade["entry"])
    risk = float(trade["risk"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp1_frac = float(trade["tp1_fraction"])
    tp2_frac = float(trade["tp2_fraction"])
    trail_mult = float(trade["trail_mult"])
    atr_period = int(trade["atr_period"])
    ema_fast_n = int(trade["ema_fast_n"])

    df = df.copy()
    df["atr"] = atr(df, atr_period)
    df["ema_fast"] = ema(df["close"], ema_fast_n)

    last_checked = int(trade["last_checked_ts"])
    df2 = df[df["ts"] > last_checked].copy()
    if df2.empty:
        return None

    trail_activation = trade.get("trail_activation", "after_tp1")
    require_trend_valid = bool(trade.get("trail_require_trend_valid", True))
    be_after_tp1 = bool(trade.get("breakeven_after_tp1", True))
    be_offset_r = float(trade.get("breakeven_offset_r", 0.0))

    for _, row in df2.iterrows():
        c = {
            "ts": int(row["ts"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        atrv = float(row["atr"]) if math.isfinite(float(row["atr"])) else None
        ema_fast_v = float(row["ema_fast"]) if math.isfinite(float(row["ema_fast"])) else None

        # Update extremes
        if side == "LONG":
            trade["highest_high"] = max(float(trade["highest_high"]), c["high"])
        else:
            trade["lowest_low"] = min(float(trade["lowest_low"]), c["low"])

        current_stop = float(trade["current_stop"])
        tp1_hit = bool(trade["tp1_hit"])

        # Decide if trailing is active
        trail_active = (trail_activation == "immediate") or tp1_hit

        # Optional: update trailing stop (chandelier) only when trend is valid (EMA_fast)
        if trail_active and atrv and atrv > 0:
            trend_ok = True
            if require_trend_valid and ema_fast_v is not None:
                trend_ok = (c["close"] >= ema_fast_v) if side == "LONG" else (c["close"] <= ema_fast_v)

            if trend_ok:
                if side == "LONG":
                    new_stop = float(trade["highest_high"]) - trail_mult * atrv
                    trade["current_stop"] = max(float(trade["current_stop"]), new_stop)
                else:
                    new_stop = float(trade["lowest_low"]) + trail_mult * atrv
                    trade["current_stop"] = min(float(trade["current_stop"]), new_stop)

            current_stop = float(trade["current_stop"])

        # Hit tests (worst-case: stop first on collision)
        stop_hit = candle_hits(side, c, current_stop, "stop")
        tp1_hit_now = candle_hits(side, c, tp1, "tp")
        tp2_hit_now = candle_hits(side, c, tp2, "tp")

        if not tp1_hit:
            if stop_hit and tp1_hit_now:
                # worst-case: stop first
                total_r = r_for_exit(side, entry, current_stop, risk)
                return (current_stop, c["ts"], "STOP_COLLISION_BEFORE_TP1", total_r)

            if stop_hit:
                total_r = r_for_exit(side, entry, current_stop, risk)
                return (current_stop, c["ts"], "STOP_BEFORE_TP1", total_r)

            if tp1_hit_now:
                trade["tp1_hit"] = True
                trade["_tp1_r_cache"] = r_for_exit(side, entry, tp1, risk)

                # breakeven move
                if be_after_tp1:
                    be_stop = entry + (be_offset_r * risk) if side == "LONG" else entry - (be_offset_r * risk)
                    if side == "LONG":
                        trade["current_stop"] = max(float(trade["current_stop"]), be_stop)
                    else:
                        trade["current_stop"] = min(float(trade["current_stop"]), be_stop)

        else:
            # After TP1: stop first (worst-case), then TP2
            if stop_hit and tp2_hit_now:
                rem_r = r_for_exit(side, entry, current_stop, risk)
                tp1_r = float(trade.get("_tp1_r_cache") or r_for_exit(side, entry, tp1, risk))
                total_r = tp1_frac * tp1_r + tp2_frac * rem_r
                return (current_stop, c["ts"], "STOP_COLLISION_AFTER_TP1", total_r)

            if stop_hit:
                rem_r = r_for_exit(side, entry, current_stop, risk)
                tp1_r = float(trade.get("_tp1_r_cache") or r_for_exit(side, entry, tp1, risk))
                total_r = tp1_frac * tp1_r + tp2_frac * rem_r
                return (current_stop, c["ts"], "TRAIL_STOP", total_r)

            if tp2_hit_now:
                rem_r = r_for_exit(side, entry, tp2, risk)
                tp1_r = float(trade.get("_tp1_r_cache") or r_for_exit(side, entry, tp1, risk))
                total_r = tp1_frac * tp1_r + tp2_frac * rem_r
                return (tp2, c["ts"], "TP2", total_r)

        trade["last_checked_ts"] = c["ts"]

    return None

def update_open_trades(r: Redis, ex: ccxt.Exchange, cfg: dict, sk: str) -> int:
    if not cfg.get("performance", {}).get("enable", True):
        return 0

    ids = list(r.smembers(k_open_set(sk)))
    if not ids:
        return 0

    closed = 0
    limit = int(cfg["strategy"]["candles_limit"])
    keep = int(cfg.get("performance", {}).get("closed_trades_keep", 20000))

    for sid in ids:
        raw = r.get(k_open_trade(sk, sid))
        if not raw:
            r.srem(k_open_set(sk), sid)
            continue

        try:
            trade = json.loads(raw)
        except Exception:
            r.delete(k_open_trade(sk, sid))
            r.srem(k_open_set(sk), sid)
            continue

        symbol = trade["symbol"]
        tf = trade["timeframe"]

        try:
            df = fetch_ohlcv_ccxt(ex, symbol, tf, limit=limit)
            closed_ts = latest_closed_candle_ts(df, tf)
            df_eval = df[df["ts"] <= closed_ts].copy()
        except Exception as e:
            print(f"[WARN] trade update fetch failed {symbol} {tf}: {repr(e)}")
            continue

        res = update_trade_with_candles(trade, df_eval)
        if res is None:
            r.set(k_open_trade(sk, sid), json.dumps(trade), ex=30 * 24 * 3600)
            continue

        exit_price, exit_ts, reason, total_r = res
        close_trade(r, sk, trade, exit_price=exit_price, exit_ts=exit_ts, reason=reason, total_r=total_r, keep=keep)
        closed += 1

    return closed


# =========================
# Main cron run
# =========================

def run_once(config_path: str = "candlepilot.config.yaml") -> None:
    cfg = load_config(config_path)
    sk = strategy_key(cfg)

    redis_url = env_required(cfg["redis"]["url_env"])
    r = Redis.from_url(redis_url, decode_responses=True, socket_timeout=10)

    lock_key = cfg["app"]["lock_key"]
    lock_ttl = int(cfg["app"]["lock_ttl_seconds"])
    if not acquire_lock(r, k_lock(lock_key), lock_ttl):
        print("[INFO] lock not acquired; exiting")
        return

    webhook_url = env_required(cfg["discord"]["webhook_url_env"])
    discord_username = cfg["discord"].get("username", "")
    discord_avatar = cfg["discord"].get("avatar_url", "")

    exchange_name = cfg["market"]["exchange"]
    market = cfg["market"]["market_type"]
    tfs: List[str] = cfg["market"]["timeframes"]

    limit = int(cfg["strategy"]["candles_limit"])
    min_conf = float(cfg["strategy"]["min_confidence"])
    pub_ttl_days = int(cfg["publishing"]["published_ttl_days"])
    cooldown_candles = int(cfg.get("publishing", {}).get("cooldown_candles", 0))

    ex = build_mexc_client(cfg)
    try:
        ex.load_markets()
    except Exception as e:
        print(f"[WARN] load_markets failed: {repr(e)} (continuing)")

    # 1) update open trades first
    closed_now = update_open_trades(r, ex, cfg, sk)

    # universe
    if cfg.get("universe", {}).get("enable", False):
        symbols = get_universe_symbols_cached(r, ex, cfg, exchange_name, market)
    else:
        symbols = cfg["market"].get("symbols", []) or []
    scan_symbols = shard_symbols(r, symbols, cfg)

    posted = 0
    scanned = 0
    errors = 0
    signals_created = 0

    for tf in tfs:
        for symbol in scan_symbols:
            scanned += 1
            try:
                df = fetch_ohlcv_ccxt(ex, symbol, tf, limit=limit)
                closed_ts = latest_closed_candle_ts(df, tf)
                if already_processed_candle(r, market, symbol, tf, closed_ts):
                    continue
                df_eval = df[df["ts"] <= closed_ts].copy()

                sig = build_trend_pullback_signal(cfg, df_eval, symbol, tf, exchange_name, market)
                if not sig:
                    continue
                if float(sig["confidence"]) < min_conf:
                    continue

                # cooldown
                if cooldown_candles > 0:
                    cd_seconds = cooldown_candles * timeframe_to_seconds(tf)
                    cd_key = cooldown_key(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"], sig["setup"], sk)
                    if in_cooldown(r, cd_key):
                        continue
                    set_cooldown(r, cd_key, cd_seconds)

                write_signal_to_redis(r, sig)
                if cfg.get("performance", {}).get("enable", True):
                    create_open_trade(r, sk, sig, cfg)
                signals_created += 1

                if mark_published(r, sig["signal_id"], pub_ttl_days):
                    post_to_discord(webhook_url, format_discord_message(sig), username=discord_username, avatar_url=discord_avatar)
                    posted += 1

            except Exception as e:
                errors += 1
                print(f"[ERROR] {symbol} {tf}: {repr(e)}")

    m = get_metrics(r, sk)
    win_rate = (m["wins"] / m["trades"]) if m["trades"] else 0.0
    avg_r = (m["sum_r"] / m["trades"]) if m["trades"] else 0.0

    print(
        f"[DONE] strategy={sk} universe={len(symbols)} scanned={scanned} signals_created={signals_created} "
        f"posted={posted} closed_trades={closed_now} errors={errors} | "
        f"perf: trades={m['trades']} winrate={win_rate:.2%} avgR={avg_r:.3f} "
        f"equityR={m['equity_r']:.3f} maxDD={m['max_drawdown_r']:.3f}"
    )


if __name__ == "__main__":
    run_once("candlepilot.config.yaml")
