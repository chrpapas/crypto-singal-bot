#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perf_common.py — shared helpers for performance scripts

Uses the same config file as mexc_movers_bot.py, reading:
- persistence.redis_url / REDIS_URL
- persistence.key_prefix
- discord.performance_webhook / DISCORD_PERFORMANCE_WEBHOOK
  (fallback to signals_webhook / DISCORD_SIGNALS_WEBHOOK)
"""

import os, json, yaml, requests, redis
import pandas as pd

_redis_client = None
_redis_prefix = "spideybot:v1"
_performance_hook = None


# =============== Config ===============
def _expand_env(o):
    if isinstance(o, dict):
        return {k: _expand_env(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_expand_env(x) for x in o]
    if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
        return os.environ.get(o[2:-1], o)
    return o


def load_config(path: str) -> dict:
    print(f"[perf_common] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_env(cfg)
    return cfg


# =============== Redis helpers ===============
def init_redis_from_config(cfg: dict):
    """Initialise global Redis client using same settings as movers bot."""
    global _redis_client, _redis_prefix

    p = cfg.get("persistence", {}) or {}
    url = p.get("redis_url") or os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("Redis URL missing. Set persistence.redis_url or REDIS_URL.")

    _redis_prefix = p.get("key_prefix", "spideybot:v1")

    _redis_client = redis.Redis.from_url(
        url, decode_responses=True, socket_timeout=6
    )
    # quick self-test
    test_key = f"{_redis_prefix}:perf_selftest"
    _redis_client.setex(test_key, 60, "ok")
    print(f"[perf_common] Redis OK | prefix={_redis_prefix}")

    return _redis_client, _redis_prefix


def _redis_key(*parts) -> str:
    return ":".join([_redis_prefix, *[str(p) for p in parts]])


def redis_load_json(*parts, default=None):
    """Load JSON-encoded object from Redis."""
    if _redis_client is None:
        raise RuntimeError("Redis client not initialised. Call init_redis_from_config(cfg) first.")
    key = _redis_key(*parts)
    txt = _redis_client.get(key)
    if not txt:
        return default if default is not None else {}
    try:
        return json.loads(txt)
    except Exception:
        return default if default is not None else {}


def load_trades(source: str = "movers"):
    """
    Load closed trades from silent registry in Redis.
    By default, only `source == "movers"` (your Movers bot).
    """
    book = redis_load_json("state", "silent_open", default={})
    trades = []
    for tr in book.values():
        if source and tr.get("source") != source:
            continue
        if tr.get("status") != "closed":
            continue
        trades.append(tr)

    print(f"[perf_common] Loaded {len(trades)} closed {source or 'all'} trades from Redis.")
    return trades


# =============== Discord helpers ===============
def init_discord_from_config(cfg: dict) -> str:
    """
    Decide which webhook to use for performance posts:
    1) DISCORD_PERFORMANCE_WEBHOOK (env)
    2) discord.performance_webhook (config)
    3) DISCORD_SIGNALS_WEBHOOK (env)
    4) discord.signals_webhook (config)
    """
    global _performance_hook

    d = cfg.get("discord", {}) or {}

    hook = (
        os.environ.get("DISCORD_PERFORMANCE_WEBHOOK")
        or d.get("performance_webhook")
        or os.environ.get("DISCORD_SIGNALS_WEBHOOK")
        or d.get("signals_webhook")
        or ""
    )

    _performance_hook = hook
    if hook:
        print("[perf_common] Using Discord webhook from env/config.")
    else:
        print("[perf_common] WARNING: No Discord webhook configured for performance.")
    return hook


def post_discord(hook: str, text: str):
    """Post a message to Discord; log errors but don't crash."""
    if not hook:
        print("[perf_common] No Discord webhook provided; skipping Discord post.")
        return
    if not text.strip():
        print("[perf_common] Empty message; not posting to Discord.")
        return
    try:
        resp = requests.post(hook, json={"content": text}, timeout=10)
        if resp.status_code >= 300:
            print(f"[perf_common] Discord post failed: HTTP {resp.status_code} {resp.text}")
        else:
            print("[perf_common] Discord post OK.")
    except Exception as e:
        print(f"[perf_common] Discord post error: {e}")
