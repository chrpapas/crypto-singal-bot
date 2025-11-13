#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perf_common.py — shared helpers for Movers performance scripts

- Loads YAML config
- Initializes Redis (same keys as mexc_movers_bot)
- Initializes Discord webhook (separate performance webhook if provided)
- Loads closed Movers trades from Redis
- Posts to Discord with automatic chunking under 2000 chars
"""

import os
import json
from typing import Dict, Any, List, Optional

import yaml
import redis
import requests
import pandas as pd

# Globals initialized by init_redis_from_config / init_discord_from_config
_redis_client: Optional[redis.Redis] = None
_redis_prefix: str = "spideybot:v1"
_discord_webhook: Optional[str] = None

# Discord hard limit is 2000 chars; keep some margin
MAX_DISCORD_LEN = 1900


# ======================================================================
# Config helpers
# ======================================================================

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file (same style as mexc_movers_bot_config)."""
    print(f"[perf_common] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# ======================================================================
# Redis helpers
# ======================================================================

def init_redis_from_config(cfg: Dict[str, Any]):
    """
    Initialize global Redis client from config:
      persistence.redis_url / REDIS_URL
      persistence.key_prefix
    """
    global _redis_client, _redis_prefix

    pers = cfg.get("persistence", {}) or {}
    url = pers.get("redis_url") or os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError(
            "[perf_common] No Redis URL. "
            "Set persistence.redis_url in config or REDIS_URL env."
        )

    prefix = pers.get("key_prefix", "spideybot:v1")
    ttl_minutes = int(pers.get("ttl_minutes", 2880))
    ttl_seconds = ttl_minutes * 60

    r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
    # simple self-test
    r.setex(f"{prefix}:perf_selftest", ttl_seconds, pd.Timestamp.utcnow().isoformat())

    _redis_client = r
    _redis_prefix = prefix

    print(f"[perf_common] Redis OK | prefix={_redis_prefix}")
    return r, prefix


def _redis_key(*parts) -> str:
    return ":".join([_redis_prefix, *[str(p) for p in parts]])


def redis_load_json(*parts, default=None):
    """Load a JSON object from Redis under the composed key."""
    if _redis_client is None:
        raise RuntimeError(
            "[perf_common] Redis not initialized. "
            "Call init_redis_from_config(cfg) first."
        )
    key = _redis_key(*parts)
    txt = _redis_client.get(key)
    if not txt:
        return default if default is not None else {}
    try:
        return json.loads(txt)
    except Exception:
        return default if default is not None else {}


# ======================================================================
# Discord helpers
# ======================================================================

def init_discord_from_config(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Initialize Discord webhook.

    Priority:
      1) DISCORD_PERFORMANCE_WEBHOOK env
      2) discord.performance_webhook in config
      3) DISCORD_SIGNALS_WEBHOOK env
      4) discord.signals_webhook in config
    """
    global _discord_webhook

    dcfg = cfg.get("discord", {}) or {}

    hook = (
        os.environ.get("DISCORD_PERFORMANCE_WEBHOOK")
        or dcfg.get("performance_webhook")
        or os.environ.get("DISCORD_SIGNALS_WEBHOOK")
        or dcfg.get("signals_webhook")
    )

    if not hook:
        print(
            "[perf_common] No Discord webhook found in env or config. "
            "Stats will not be posted."
        )
    else:
        print("[perf_common] Using Discord webhook from env/config.")

    _discord_webhook = hook
    return hook


def post_discord(hook: Optional[str], content: str):
    """
    Post to Discord, automatically chunking messages that exceed the
    2000-character limit (we use MAX_DISCORD_LEN for safety).

    Signature kept as post_discord(hook, content) to match existing scripts.
    If hook is None, falls back to the globally initialized webhook.
    """
    global _discord_webhook

    if hook is None:
        hook = _discord_webhook

    if not hook:
        print("[perf_common] No Discord webhook configured; skipping post.")
        return

    if not content or not content.strip():
        print("[perf_common] Empty content; skipping post.")
        return

    # Split on newlines so chunks stay readable
    lines = content.split("\n")
    chunks: List[str] = []
    current = ""

    for line in lines:
        # length if we add this line (plus newline if needed)
        extra_len = len(line) + (1 if current else 0)
        if len(current) + extra_len > MAX_DISCORD_LEN:
            # finalize current chunk
            chunks.append(current)
            current = line
        else:
            if current:
                current += "\n" + line
            else:
                current = line

    if current:
        chunks.append(current)

    # Send all chunks
    for idx, chunk in enumerate(chunks, start=1):
        try:
            resp = requests.post(hook, json={"content": chunk}, timeout=10)
            if resp.status_code >= 400:
                print(
                    f"[perf_common] Discord post failed "
                    f"(chunk {idx}/{len(chunks)}): HTTP {resp.status_code} {resp.text}"
                )
            else:
                print(
                    f"[perf_common] Discord post OK "
                    f"(chunk {idx}/{len(chunks)}) len={len(chunk)}"
                )
        except Exception as e:
            print(
                f"[perf_common] Discord post error "
                f"(chunk {idx}/{len(chunks)}): {e}"
            )


# ======================================================================
# Trades loader
# ======================================================================

def load_trades() -> List[Dict[str, Any]]:
    """
    Load *closed movers trades* from the same Redis layout that
    mexc_movers_bot.py uses (state:silent_open).
    """
    book = redis_load_json("state", "silent_open", default={})
    trades = [
        tr
        for tr in book.values()
        if tr.get("source") == "movers" and tr.get("status") == "closed"
    ]

    print(
        f"[perf_common] Loaded {len(trades)} closed movers trades from Redis."
    )
    return trades
