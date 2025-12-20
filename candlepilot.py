import os
import time
import json
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import httpx
import numpy as np
import pandas as pd
from redis import Redis


# ----------------------------
# Config + helpers
# ----------------------------

def load_config(path: str = "config.yaml") -> dict:
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
    # Supports "1m", "5m", "15m", "1h", "4h", "1d"
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
    # Basic rounding to reduce noisy decimals (you can customize per symbol tick size later)
    if x == 0 or not math.isfinite(x):
        return x
    # keep ~6 significant digits
    digits = max(0, 6 - int(math.floor(math.log10(abs(x)))) - 1)
    return round(x, digits)


# ----------------------------
# Redis keys
# ----------------------------

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


# ----------------------------
# Indicators
# ----------------------------

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


# ----------------------------
# MEXC PERP OHLCV fetch (public)
# ----------------------------
# NOTE:
# MEXC contract/perps endpoints differ across versions.
# This function is isolated so you can easily swap it with your existing MEXC client call.
#
# Expected output df columns:
#   ts (int, seconds), open, high, low, close, volume (floats)
#
# If this endpoint doesn't match your current integration, replace fetch_mexc_perp_ohlcv()
# to call your existing MEXC API wrapper and return the same DataFrame format.

MEXC_HTTP_TIMEOUT = 15.0

def mexc_tf_to_interval(tf: str) -> str:
    # MEXC typically uses "Min1", "Min15", "Hour1" etc for some endpoints,
    # others use "1m", "15m", "1h". Adjust as needed.
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]

async def fetch_mexc_perp_ohlcv(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    """
    Tries a common style of MEXC contract kline endpoint.
    If it fails in your environment, replace this with your own MEXC client.
    """
    interval = mexc_tf_to_interval(tf)

    # Common patterns seen in MEXC contract APIs:
    # - https://contract.mexc.com/api/v1/contract/kline/{symbol}?interval=...&limit=...
    # - https://contract.mexc.com/api/v1/contract/kline/{symbol}?interval=...&start=...&end=...
    #
    # We'll attempt a straightforward call:
    base = "https://contract.mexc.com"
    url = f"{base}/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": limit}

    async with httpx.AsyncClient(timeout=MEXC_HTTP_TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    # Expected data formats vary.
    # We'll handle a couple of typical shapes.

    # Shape A: {"success":true,"data":[[ts,open,high,low,close,vol],...]}
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and data["data"] and isinstance(data["data"][0], list):
        rows = data["data"]
        # ts may be ms; normalize to seconds if needed
        out = []
        for r in rows:
            ts = int(r[0])
            if ts > 10_000_000_000:  # ms
                ts //= 1000
            out.append({
                "ts": ts,
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]) if len(r) > 5 else 0.0,
            })
        df = pd.DataFrame(out)

    # Shape B: {"data":{"time":[...],"open":[...],...}}
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], dict) and "time" in data["data"]:
        d = data["data"]
        out = []
        for i in range(len(d["time"])):
            ts = int(d["time"][i])
            if ts > 10_000_000_000:
                ts //= 1000
            out.append({
                "ts": ts,
                "open": float(d["open"][i]),
                "high": float(d["high"][i]),
                "low": float(d["low"][i]),
                "close": float(d["close"][i]),
                "volume": float(d["vol"][i]) if "vol" in d else float(d.get("volume", [0]*len(d["time"]))[i]),
            })
        df = pd.DataFrame(out)

    else:
        raise RuntimeError(f"Unrecognized MEXC kline response shape: {str(data)[:300]}")

    df = df.dropna().sort_values("ts").reset_index(drop=True)
    return df


# ----------------------------
# Strategy: Trend Pullback v1
# ----------------------------

@dataclass
class Signal:
    signal_id: str
    payload: Dict[str, Any]

def compute_confidence(df: pd.DataFrame, side: str, ema_trend_col: str, atr_col: str) -> float:
    """
    Simple, stable confidence:
    - More distance from EMA200 in direction of trend (normalized by ATR) increases confidence.
    - Too close to EMA200 reduces confidence (chop risk).
    """
    last = df.iloc[-1]
    atrv = float(last[atr_col])
    if not math.isfinite(atrv) or atrv <= 0:
        return 0.0
    dist = (float(last["close"]) - float(last[ema_trend_col])) / atrv
    # For shorts, invert dist
    if side == "SHORT":
        dist = -dist
    # Map dist roughly to [0,1] with a soft cap
    # dist ~ 0.5 => ok, dist ~ 2 => strong
    score = 1 / (1 + math.exp(-(dist - 0.5)))
    return float(np.clip(score, 0.0, 1.0))

def build_trend_pullback_signal(
    cfg: dict,
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    exchange: str,
    market: str,
) -> Optional[Signal]:
    s_cfg = cfg["strategy"]
    ema_fast_n = int(s_cfg["ema_fast"])
    ema_trend_n = int(s_cfg["ema_trend"])
    atr_n = int(s_cfg["atr_period"])

    # Need enough data
    if len(df) < max(ema_trend_n, atr_n) + 5:
        return None

    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast_n)
    df["ema_trend"] = ema(df["close"], ema_trend_n)
    df["atr"] = atr(df, atr_n)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Only signal on CLOSED candle
    candle_ts = int(last["ts"])
    if not is_candle_closed(candle_ts, tf):
        return None

    close = float(last["close"])
    ema_fast_v = float(last["ema_fast"])
    ema_trend_v = float(last["ema_trend"])
    atr_v = float(last["atr"])

    if not (math.isfinite(atr_v) and atr_v > 0):
        return None

    trend_up = close > ema_trend_v
    trend_down = close < ema_trend_v

    # Pullback + reclaim EMA_fast
    pullback_long = (
        trend_up
        and float(prev["close"]) < float(prev["ema_fast"])
        and close > ema_fast_v
    )
    pullback_short = (
        trend_down
        and float(prev["close"]) > float(prev["ema_fast"])
        and close < ema_fast_v
    )

    if not (pullback_long or pullback_short):
        return None

    side = "LONG" if pullback_long else "SHORT"

    # Entry model: signal candle close (deterministic, easy)
    entry = close

    stop_mult = float(s_cfg["stop_atr_mult"])
    tp1_r = float(s_cfg["tp1_r"])
    tp2_r = float(s_cfg["tp2_r"])
    trail_mult = float(s_cfg["trail_atr_mult"])

    stop = entry - stop_mult * atr_v if side == "LONG" else entry + stop_mult * atr_v
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    tp1 = entry + tp1_r * risk if side == "LONG" else entry - tp1_r * risk
    tp2 = entry + tp2_r * risk if side == "LONG" else entry - tp2_r * risk

    confidence = compute_confidence(df, side, "ema_trend", "atr")

    # signal_id deterministic (so we can prove “no edits” later)
    raw_id = json.dumps({
        "exchange": exchange,
        "market": market,
        "symbol": symbol,
        "tf": tf,
        "candle_ts": candle_ts,
        "setup": s_cfg["name"],
        "side": side,
    }, sort_keys=True)
    signal_id = sha256_hex(raw_id)[:24]

    payload = {
        "signal_id": signal_id,
        "ts": candle_ts,
        "exchange": exchange,
        "market": market,
        "symbol": symbol,
        "timeframe": tf,
        "setup": s_cfg["name"],
        "side": side,
        "confidence": round(float(confidence), 4),
        "entry_model": "signal_close",
        "entry": {"type": "market", "price": round_price(entry)},
        "exit": {
            "stop": round_price(stop),
            "take_profit": [round_price(tp1), round_price(tp2)],
            "trail": {
                "method": "atr_chandelier",
                "mult": trail_mult,
                "activation": "after_tp1" if bool(s_cfg.get("trail_activate_after_tp1", True)) else "immediate",
            },
        },
        "invalidation": f"Close crosses EMA{int(s_cfg['ema_trend'])} against direction",
        "rationale": [
            f"Trend filter: close vs EMA{int(s_cfg['ema_trend'])}",
            f"Pullback reclaim EMA{int(s_cfg['ema_fast'])}",
            f"ATR({int(s_cfg['atr_period'])}) risk model",
        ],
    }

    return Signal(signal_id=signal_id, payload=payload)


# ----------------------------
# Discord publishing
# ----------------------------

def format_discord_message(sig: Dict[str, Any]) -> str:
    side = sig["side"]
    sym = sig["symbol"]
    tf = sig["timeframe"]
    conf = sig["confidence"]
    entry = sig["entry"]["price"]
    stop = sig["exit"]["stop"]
    tp1, tp2 = sig["exit"]["take_profit"]

    # quick RR display
    risk = abs(entry - stop)
    r1 = abs(tp1 - entry) / risk if risk > 0 else 0
    r2 = abs(tp2 - entry) / risk if risk > 0 else 0

    return (
        f"**{side}** `{sym}`  •  **{tf}**  •  conf: **{conf}**\n"
        f"Entry: `{entry}`\n"
        f"Stop: `{stop}`\n"
        f"TP1: `{tp1}`  (≈ {r1:.2f}R)\n"
        f"TP2: `{tp2}`  (≈ {r2:.2f}R)\n"
        f"Invalidation: _{sig['invalidation']}_\n"
        f"Signal ID: `{sig['signal_id']}`"
    )

async def post_to_discord(webhook_url: str, content: str, username: str = "", avatar_url: str = "") -> None:
    payload = {"content": content}
    if username:
        payload["username"] = username
    if avatar_url:
        payload["avatar_url"] = avatar_url

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(webhook_url, json=payload)
        resp.raise_for_status()


# ----------------------------
# Main cron run
# ----------------------------

def acquire_lock(r: Redis, key: str, ttl: int) -> bool:
    # True if lock acquired
    return bool(r.set(key, "1", nx=True, ex=ttl))

def already_processed_candle(r: Redis, market: str, symbol: str, tf: str, candle_ts: int) -> bool:
    key = k_last_candle(market, symbol, tf)
    last = r.get(key)
    if last is not None and int(last) >= candle_ts:
        return True
    r.set(key, str(candle_ts), ex=14 * 24 * 3600)  # keep 2 weeks
    return False

def write_signal_to_redis(r: Redis, sig: Dict[str, Any]) -> None:
    key_latest = k_latest(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"])
    key_hist = k_history(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"])
    r.set(key_latest, json.dumps(sig), ex=6 * 3600)  # 6h
    r.lpush(key_hist, json.dumps(sig))
    r.ltrim(key_hist, 0, 200)

def mark_published(r: Redis, signal_id: str, ttl_days: int) -> bool:
    # returns True if newly marked (not published before)
    return bool(r.set(k_published(signal_id), "1", nx=True, ex=ttl_days * 86400))

async def run_once() -> None:
    cfg = load_config("config.yaml")

    redis_url = env_required(cfg["redis"]["url_env"])
    r = Redis.from_url(redis_url, decode_responses=True, socket_timeout=10)

    # lock
    lock_key = cfg["app"]["lock_key"]
    lock_ttl = int(cfg["app"]["lock_ttl_seconds"])
    if not acquire_lock(r, k_lock(lock_key), lock_ttl):
        return

    # Discord webhook
    webhook_url = env_required(cfg["discord"]["webhook_url_env"])
    discord_username = cfg["discord"].get("username", "")
    discord_avatar = cfg["discord"].get("avatar_url", "")

    exchange = cfg["market"]["exchange"]
    market = cfg["market"]["market_type"]
    symbols = cfg["market"]["symbols"]
    tfs = cfg["market"]["timeframes"]

    limit = int(cfg["strategy"]["candles_limit"])
    min_conf = float(cfg["strategy"]["min_confidence"])
    pub_ttl_days = int(cfg["publishing"]["published_ttl_days"])

    posted_count = 0

    for tf in tfs:
        for symbol in symbols:
            try:
                # Fetch candles (perps)
                df = await fetch_mexc_perp_ohlcv(symbol, tf, limit=limit)

                # Ensure newest candle is closed before generating signals
                last_ts = int(df.iloc[-1]["ts"])
                if not is_candle_closed(last_ts, tf):
                    # If last isn't closed, try previous (closed) candle for gating.
                    # We still compute signal on df as-is, but build_signal checks close anyway.
                    pass

                # Dedupe per (symbol, tf, candle)
                # Only proceed if candle advanced
                if already_processed_candle(r, market, symbol, tf, last_ts):
                    # Might still produce a signal on a prior closed candle;
                    # but to keep MVP simple, we gate on the latest candle timestamp.
                    continue

                sig_obj = build_trend_pullback_signal(cfg, df, symbol, tf, exchange, market)
                if not sig_obj:
                    continue

                sig = sig_obj.payload
                if float(sig["confidence"]) < min_conf:
                    continue

                # Store in Redis
                write_signal_to_redis(r, sig)

                # Publish to Discord (dedupe by signal_id)
                if mark_published(r, sig["signal_id"], pub_ttl_days):
                    msg = format_discord_message(sig)
                    await post_to_discord(webhook_url, msg, username=discord_username, avatar_url=discord_avatar)
                    posted_count += 1

            except Exception as e:
                # Keep cron robust; log and continue
                print(f"[ERROR] {symbol} {tf}: {repr(e)}")

    print(f"[DONE] posted={posted_count}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_once())
