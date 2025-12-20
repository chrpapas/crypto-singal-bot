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


# =========================
# Redis keys + primitives
# =========================

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

def acquire_lock(r: Redis, key: str, ttl: int) -> bool:
    return bool(r.set(key, "1", nx=True, ex=ttl))

def already_processed_candle(r: Redis, market: str, symbol: str, tf: str, candle_ts: int) -> bool:
    """
    Ensures we only process each (symbol, tf, latest candle ts) once.
    """
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

def cooldown_key(exchange: str, market: str, symbol: str, tf: str, setup: str) -> str:
    return f"cooldown:{exchange}:{market}:{symbol}:{tf}:{setup}"

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
# Discord (webhook)
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

    return (
        f"**{setup}** | **{side}** `{sym}` • **{tf}** • conf: **{conf}**\n"
        f"Entry (model: {sig.get('entry_model','signal_close')}): `{entry}`\n"
        f"Stop: `{stop}`\n"
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
# CCXT: MEXC PERP data
# =========================

def build_mexc_client(cfg: dict) -> ccxt.Exchange:
    mexc_cfg = cfg.get("mexc", {})
    key_env = mexc_cfg.get("api_key_env", "MEXC_KEY")
    sec_env = mexc_cfg.get("api_secret_env", "MEXC_SECRET")

    api_key = os.environ.get(key_env, "")
    api_secret = os.environ.get(sec_env, "")

    ex = ccxt.mexc({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # IMPORTANT for perps
        },
    })
    return ex

def normalize_symbol_for_ccxt(symbol: str) -> str:
    """
    Accepts:
    - "BTC/USDT:USDT" (ideal for perps)
    - "BTC_USDT" or "BTCUSDT" -> heuristics to "BTC/USDT:USDT"
    """
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

    df = pd.DataFrame(out).dropna().sort_values("ts").reset_index(drop=True)
    return df


# =========================
# Universe (broad scan)
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
        if not m:
            continue
        if not m.get("active", True):
            continue

        # swaps/perps only
        if m.get("type") != "swap" and not m.get("swap", False):
            continue

        # quote filter
        if (m.get("quote") or "").upper() != quote:
            continue

        # skip inverse if flagged
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

    # Rank by quote volume
    tickers = {}
    try:
        # some exchanges allow passing a list; if it fails, fall back to full tickers
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
    """
    Returns a rotating slice of the universe so each cron run scans only a subset.
    """
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

    # current index in Redis
    idx_raw = r.get(state_key)
    idx = int(idx_raw) if idx_raw is not None else 0
    idx = idx % n

    # slice with wrap-around
    end = idx + per_run
    if end <= n:
        chunk = symbols[idx:end]
    else:
        chunk = symbols[idx:] + symbols[:(end % n)]

    # advance index
    next_idx = (idx + step) % n
    r.set(state_key, str(next_idx), ex=30 * 24 * 3600)

    return chunk


# =========================
# Strategy: Trend Pullback v1
# =========================

def compute_confidence(df: pd.DataFrame, side: str) -> float:
    last = df.iloc[-1]
    atrv = float(last["atr"])
    if not math.isfinite(atrv) or atrv <= 0:
        return 0.0

    dist = (float(last["close"]) - float(last["ema_trend"])) / atrv
    if side == "SHORT":
        dist = -dist

    # dist ~ 0.5 ok, dist ~ 2 strong
    score = 1 / (1 + math.exp(-(dist - 0.5)))
    return float(np.clip(score, 0.0, 1.0))

def build_trend_pullback_signal(
    cfg: dict,
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    exchange: str,
    market: str,
) -> Optional[Dict[str, Any]]:
    s_cfg = cfg["strategy"]

    ema_fast_n = int(s_cfg["ema_fast"])
    ema_trend_n = int(s_cfg["ema_trend"])
    atr_n = int(s_cfg["atr_period"])

    if len(df) < max(ema_trend_n, atr_n) + 5:
        return None

    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast_n)
    df["ema_trend"] = ema(df["close"], ema_trend_n)
    df["atr"] = atr(df, atr_n)

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

    trend_up = close > ema_trend_v
    trend_down = close < ema_trend_v

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
    entry = close  # deterministic model (signal close)

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

    confidence = compute_confidence(df, side)

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

    sig = {
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
        "invalidation": f"Close crosses EMA{ema_trend_n} against direction",
        "rationale": [
            f"Trend filter: close vs EMA{ema_trend_n}",
            f"Pullback reclaim EMA{ema_fast_n}",
            f"ATR({atr_n}) risk model",
        ],
    }
    return sig


# =========================
# Main cron run (sync)
# =========================

def run_once(config_path: str = "candlepilot.config.yaml") -> None:
    cfg = load_config(config_path)

    # Redis
    redis_url = env_required(cfg["redis"]["url_env"])
    r = Redis.from_url(redis_url, decode_responses=True, socket_timeout=10)

    # Lock
    lock_key = cfg["app"]["lock_key"]
    lock_ttl = int(cfg["app"]["lock_ttl_seconds"])
    if not acquire_lock(r, k_lock(lock_key), lock_ttl):
        print("[INFO] lock not acquired; exiting")
        return

    # Discord
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

    # Build exchange
    ex = build_mexc_client(cfg)

    try:
        ex.load_markets()
    except Exception as e:
        print(f"[WARN] load_markets failed: {repr(e)} (continuing)")

    # Universe symbols
    symbols: List[str] = []
    if cfg.get("universe", {}).get("enable", False):
        symbols = get_universe_symbols_cached(r, ex, cfg, exchange_name, market)
    else:
        # fallback if you disable universe mode
        symbols = cfg["market"].get("symbols", []) or []

    # Shard symbols per run to avoid rate limits/timeouts
    scan_symbols = shard_symbols(r, symbols, cfg)

    posted = 0
    scanned = 0
    errors = 0

    for tf in tfs:
        for symbol in scan_symbols:
            scanned += 1
            try:
                df = fetch_ohlcv_ccxt(ex, symbol, tf, limit=limit)
                last_ts = int(df.iloc[-1]["ts"])

                # Per-(symbol,tf) candle gating
                if already_processed_candle(r, market, symbol, tf, last_ts):
                    continue

                sig = build_trend_pullback_signal(cfg, df, symbol, tf, exchange_name, market)
                if not sig:
                    continue

                if float(sig["confidence"]) < min_conf:
                    continue

                # Cooldown by candles (optional anti-spam)
                if cooldown_candles > 0:
                    cd_seconds = cooldown_candles * timeframe_to_seconds(tf)
                    cd_key = cooldown_key(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"], sig["setup"])
                    if in_cooldown(r, cd_key):
                        continue
                    set_cooldown(r, cd_key, cd_seconds)

                # Store + publish
                write_signal_to_redis(r, sig)

                if mark_published(r, sig["signal_id"], pub_ttl_days):
                    msg = format_discord_message(sig)
                    post_to_discord(webhook_url, msg, username=discord_username, avatar_url=discord_avatar)
                    posted += 1

            except Exception as e:
                errors += 1
                print(f"[ERROR] {symbol} {tf}: {repr(e)}")

    print(f"[DONE] universe={len(symbols)} scanned={scanned} posted={posted} errors={errors}")


if __name__ == "__main__":
    run_once("candlepilot.config.yaml")
