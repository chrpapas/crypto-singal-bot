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

# Performance keys
def k_open_set() -> str:
    return "trade:open:set"

def k_open_trade(signal_id: str) -> str:
    return f"trade:open:{signal_id}"

def k_closed_list() -> str:
    return "trade:closed"  # append-only list (LPUSH)

def k_metrics_global() -> str:
    return "metrics:global"


def acquire_lock(r: Redis, key: str, ttl: int) -> bool:
    return bool(r.set(key, "1", nx=True, ex=ttl))

def already_processed_candle(r: Redis, market: str, symbol: str, tf: str, candle_ts: int) -> bool:
    """
    Ensures we only process each (symbol, tf, latest CLOSED candle ts) once.
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

def latest_closed_candle_ts(df: pd.DataFrame, tf: str) -> int:
    """
    Returns open timestamp (sec) of the latest CLOSED candle.
    If the last candle is still forming, use previous candle.
    """
    if len(df) < 2:
        return int(df.iloc[-1]["ts"])
    last_ts = int(df.iloc[-1]["ts"])
    if is_candle_closed(last_ts, tf):
        return last_ts
    return int(df.iloc[-2]["ts"])


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
    idx = idx % n

    end = idx + per_run
    if end <= n:
        chunk = symbols[idx:end]
    else:
        chunk = symbols[idx:] + symbols[:(end % n)]

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
        # performance params (explicit)
        "perf": {
            "risk_r": 1.0,
            "tp1_fraction": 0.5,
            "tp2_fraction": 0.5,
        }
    }
    return sig


# =========================
# Performance tracking
# =========================

def get_metrics(r: Redis) -> Dict[str, Any]:
    raw = r.get(k_metrics_global())
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

def save_metrics(r: Redis, m: Dict[str, Any]) -> None:
    r.set(k_metrics_global(), json.dumps(m), ex=90 * 24 * 3600)

def update_metrics_on_close(r: Redis, closed_trade: Dict[str, Any]) -> None:
    m = get_metrics(r)
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
    save_metrics(r, m)

def open_trade_exists(r: Redis, signal_id: str) -> bool:
    return r.exists(k_open_trade(signal_id)) == 1

def create_open_trade(r: Redis, sig: Dict[str, Any], strategy_cfg: Dict[str, Any]) -> None:
    """
    Create an open "virtual" trade for performance tracking.
    Deterministic entry model = signal close price at sig['ts'].
    """
    sid = sig["signal_id"]
    if open_trade_exists(r, sid):
        return

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
    atr_period = int(strategy_cfg["atr_period"])

    trade = {
        "signal_id": sid,
        "exchange": sig["exchange"],
        "market": sig["market"],
        "symbol": sig["symbol"],
        "timeframe": sig["timeframe"],
        "side": side,
        "setup": sig.get("setup", ""),
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
        "atr_period": atr_period,

        "status": "OPEN",
        "tp1_hit": False,
        "current_stop": stop0,     # may trail after TP1
        "highest_high": entry,      # for long trail
        "lowest_low": entry,        # for short trail
        "last_checked_ts": int(sig["ts"]),  # process candles strictly after entry candle
    }

    pipe = r.pipeline()
    pipe.set(k_open_trade(sid), json.dumps(trade), ex=30 * 24 * 3600)
    pipe.sadd(k_open_set(), sid)
    pipe.execute()

def close_trade(r: Redis, trade: Dict[str, Any], exit_price: float, exit_ts: int, exit_reason: str, total_r: float) -> None:
    closed = {
        "signal_id": trade["signal_id"],
        "exchange": trade["exchange"],
        "market": trade["market"],
        "symbol": trade["symbol"],
        "timeframe": trade["timeframe"],
        "side": trade["side"],
        "setup": trade.get("setup", ""),
        "opened_ts": trade["opened_ts"],
        "closed_ts": exit_ts,
        "entry": trade["entry"],
        "stop0": trade["stop0"],
        "tp1": trade["tp1"],
        "tp2": trade["tp2"],
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "r_multiple": round(float(total_r), 6),
    }

    pipe = r.pipeline()
    pipe.lpush(k_closed_list(), json.dumps(closed))
    pipe.ltrim(k_closed_list(), 0, 5000)  # keep last 5000 closed trades
    pipe.delete(k_open_trade(trade["signal_id"]))
    pipe.srem(k_open_set(), trade["signal_id"])
    pipe.execute()

    update_metrics_on_close(r, closed)

def r_for_exit(side: str, entry: float, exit_price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    if side == "LONG":
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk

def candle_hits_long(candle: Dict[str, float], level: float, kind: str) -> bool:
    # kind: "stop" uses low <= level, "tp" uses high >= level
    if kind == "stop":
        return candle["low"] <= level
    return candle["high"] >= level

def candle_hits_short(candle: Dict[str, float], level: float, kind: str) -> bool:
    # short: stop if high >= level, tp if low <= level
    if kind == "stop":
        return candle["high"] >= level
    return candle["low"] <= level

def update_trade_with_candles(trade: Dict[str, Any], df: pd.DataFrame) -> Optional[Tuple[float, int, str, float]]:
    """
    Process candles after trade['last_checked_ts'] and decide if trade closes.
    Returns (exit_price, exit_ts, reason, total_r) if closed, else None.
    """

    side = trade["side"]
    entry = float(trade["entry"])
    risk = float(trade["risk"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp1_frac = float(trade["tp1_fraction"])
    tp2_frac = float(trade["tp2_fraction"])
    trail_mult = float(trade["trail_mult"])
    atr_period = int(trade["atr_period"])

    # Build ATR for trailing
    df = df.copy()
    df["atr"] = atr(df, atr_period)

    # Only consider candles strictly AFTER entry candle open ts
    last_checked = int(trade["last_checked_ts"])
    df2 = df[df["ts"] > last_checked].copy()
    if df2.empty:
        return None

    # Helpers for hit tests
    for _, row in df2.iterrows():
        c = {
            "ts": int(row["ts"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }

        # Update extremes for trailing
        if side == "LONG":
            trade["highest_high"] = max(float(trade["highest_high"]), c["high"])
        else:
            trade["lowest_low"] = min(float(trade["lowest_low"]), c["low"])

        # Determine current stop (may trail after TP1)
        current_stop = float(trade["current_stop"])
        tp1_hit = bool(trade["tp1_hit"])

        # Worst-case collision rule: if stop and tp in same candle -> stop first.
        if side == "LONG":
            stop_hit = candle_hits_long(c, current_stop, "stop")
            tp1_hit_now = candle_hits_long(c, tp1, "tp")
            tp2_hit_now = candle_hits_long(c, tp2, "tp")

            if not tp1_hit:
                # before TP1: SL or TP1 possible
                if stop_hit and tp1_hit_now:
                    # worst-case: stop first
                    total_r = -1.0
                    return (current_stop, c["ts"], "STOP_BEFORE_TP1_COLLISION", total_r)
                if stop_hit:
                    total_r = -1.0
                    return (current_stop, c["ts"], "STOP_BEFORE_TP1", total_r)
                if tp1_hit_now:
                    trade["tp1_hit"] = True
                    tp1_hit = True
                    # after TP1, trailing may start
            else:
                # after TP1: check stop first (worst-case), then TP2
                stop_hit = candle_hits_long(c, current_stop, "stop")
                tp2_hit_now = candle_hits_long(c, tp2, "tp")
                if stop_hit and tp2_hit_now:
                    # worst-case: stop first
                    rem_r = r_for_exit(side, entry, current_stop, risk)
                    total_r = tp1_frac * float(trade.get("_tp1_r_cache", r_for_exit(side, entry, tp1, risk))) + tp2_frac * rem_r
                    return (current_stop, c["ts"], "TRAIL_STOP_COLLISION", total_r)
                if stop_hit:
                    rem_r = r_for_exit(side, entry, current_stop, risk)
                    total_r = tp1_frac * float(trade.get("_tp1_r_cache", r_for_exit(side, entry, tp1, risk))) + tp2_frac * rem_r
                    return (current_stop, c["ts"], "TRAIL_STOP", total_r)
                if tp2_hit_now:
                    rem_r = r_for_exit(side, entry, tp2, risk)
                    total_r = tp1_frac * float(trade.get("_tp1_r_cache", r_for_exit(side, entry, tp1, risk))) + tp2_frac * rem_r
                    return (tp2, c["ts"], "TP2", total_r)

        else:
            # SHORT
            stop_hit = candle_hits_short(c, current_stop, "stop")
            tp1_hit_now = candle_hits_short(c, tp1, "tp")
            tp2_hit_now = candle_hits_short(c, tp2, "tp")

            if not tp1_hit:
                if stop_hit and tp1_hit_now:
                    total_r = -1.0
                    return (current_stop, c["ts"], "STOP_BEFORE_TP1_COLLISION", total_r)
                if stop_hit:
                    total_r = -1.0
                    return (current_stop, c["ts"], "STOP_BEFORE_TP1", total_r)
                if tp1_hit_now:
                    trade["tp1_hit"] = True
                    tp1_hit = True
            else:
                stop_hit = candle_hits_short(c, current_stop, "stop")
                tp2_hit_now = candle_hits_short(c, tp2, "tp")
                if stop_hit and tp2_hit_now:
                    rem_r = r_for_exit(side, entry, current_stop, risk)
                    total_r = tp1_frac * float(trade.get("_tp1_r_cache", r_for_exit(side, entry, tp1, risk))) + tp2_frac * rem_r
                    return (current_stop, c["ts"], "TRAIL_STOP_COLLISION", total_r)
                if stop_hit:
                    rem_r = r_for_exit(side, entry, current_stop, risk)
                    total_r = tp1_frac * float(trade.get("_tp1_r_cache", r_for_exit(side, entry, tp1, risk))) + tp2_frac * rem_r
                    return (current_stop, c["ts"], "TRAIL_STOP", total_r)
                if tp2_hit_now:
                    rem_r = r_for_exit(side, entry, tp2, risk)
                    total_r = tp1_frac * float(trade.get("_tp1_r_cache", r_for_exit(side, entry, tp1, risk))) + tp2_frac * rem_r
                    return (tp2, c["ts"], "TP2", total_r)

        # If TP1 got hit at some point, cache its R and update trailing stop on candle close
        if trade["tp1_hit"] and "_tp1_r_cache" not in trade:
            trade["_tp1_r_cache"] = r_for_exit(side, entry, tp1, risk)

        if trade["tp1_hit"]:
            atrv = float(row["atr"]) if math.isfinite(float(row["atr"])) else None
            if atrv and atrv > 0:
                if side == "LONG":
                    new_stop = float(trade["highest_high"]) - trail_mult * atrv
                    trade["current_stop"] = max(float(trade["current_stop"]), new_stop)
                else:
                    new_stop = float(trade["lowest_low"]) + trail_mult * atrv
                    trade["current_stop"] = min(float(trade["current_stop"]), new_stop)

        trade["last_checked_ts"] = c["ts"]

    return None

def update_open_trades(r: Redis, ex: ccxt.Exchange, cfg: dict) -> int:
    """
    Update all open trades and close those that hit exits.
    Returns number of trades closed this run.
    """
    ids = list(r.smembers(k_open_set()))
    if not ids:
        return 0

    closed = 0
    limit = int(cfg["strategy"]["candles_limit"])

    for sid in ids:
        raw = r.get(k_open_trade(sid))
        if not raw:
            r.srem(k_open_set(), sid)
            continue

        try:
            trade = json.loads(raw)
        except Exception:
            r.delete(k_open_trade(sid))
            r.srem(k_open_set(), sid)
            continue

        symbol = trade["symbol"]
        tf = trade["timeframe"]

        # Fetch candles and evaluate only closed candles up to latest closed ts
        try:
            df = fetch_ohlcv_ccxt(ex, symbol, tf, limit=limit)
            closed_ts = latest_closed_candle_ts(df, tf)
            df_eval = df[df["ts"] <= closed_ts].copy()
        except Exception as e:
            # don't kill the whole cron run
            print(f"[WARN] trade update fetch failed {symbol} {tf}: {repr(e)}")
            continue

        res = update_trade_with_candles(trade, df_eval)
        if res is None:
            # persist updated trade state (last_checked/current_stop/extremes)
            r.set(k_open_trade(sid), json.dumps(trade), ex=30 * 24 * 3600)
            continue

        exit_price, exit_ts, reason, total_r = res
        close_trade(r, trade, exit_price=exit_price, exit_ts=exit_ts, exit_reason=reason, total_r=total_r)
        closed += 1

    return closed


# =========================
# Main cron run (sync)
# =========================

def run_once(config_path: str = "candlepilot.config.yaml") -> None:
    cfg = load_config(config_path)

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

    # 1) Update open trades FIRST (performance tracking)
    closed_now = update_open_trades(r, ex, cfg)

    # Universe symbols
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

                # use latest CLOSED candle for gating
                closed_ts = latest_closed_candle_ts(df, tf)
                if already_processed_candle(r, market, symbol, tf, closed_ts):
                    continue

                df_eval = df[df["ts"] <= closed_ts].copy()

                sig = build_trend_pullback_signal(cfg, df_eval, symbol, tf, exchange_name, market)
                if not sig:
                    continue

                if float(sig["confidence"]) < min_conf:
                    continue

                # Optional anti-spam cooldown by candles
                if cooldown_candles > 0:
                    cd_seconds = cooldown_candles * timeframe_to_seconds(tf)
                    cd_key = cooldown_key(sig["exchange"], sig["market"], sig["symbol"], sig["timeframe"], sig["setup"])
                    if in_cooldown(r, cd_key):
                        continue
                    set_cooldown(r, cd_key, cd_seconds)

                # Store signal + create open trade for performance tracking
                write_signal_to_redis(r, sig)
                create_open_trade(r, sig, cfg["strategy"])
                signals_created += 1

                # Publish to Discord (dedupe)
                if mark_published(r, sig["signal_id"], pub_ttl_days):
                    msg = format_discord_message(sig)
                    post_to_discord(webhook_url, msg, username=discord_username, avatar_url=discord_avatar)
                    posted += 1

            except Exception as e:
                errors += 1
                print(f"[ERROR] {symbol} {tf}: {repr(e)}")

    m = get_metrics(r)
    win_rate = (m["wins"] / m["trades"]) if m["trades"] else 0.0
    avg_r = (m["sum_r"] / m["trades"]) if m["trades"] else 0.0

    print(
        f"[DONE] universe={len(symbols)} scanned={scanned} signals_created={signals_created} "
        f"posted={posted} closed_trades={closed_now} errors={errors} | "
        f"perf: trades={m['trades']} winrate={win_rate:.2%} avgR={avg_r:.3f} "
        f"equityR={m['equity_r']:.3f} maxDD={m['max_drawdown_r']:.3f}"
    )


if __name__ == "__main__":
    run_once("candlepilot.config.yaml")
