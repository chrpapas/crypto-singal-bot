# perf_common.py
import os
import json
import yaml
import requests
import redis
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

# ---------- Config helpers ----------

def expand_env(o):
    if isinstance(o, dict):
        return {k: expand_env(v) for k, v in o.items()}
    if isinstance(o, list):
        return [expand_env(x) for x in o]
    if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
        return os.environ.get(o[2:-1], o)
    return o

def load_config(path: str) -> Dict[str, Any]:
    print(f"[perf_common] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = expand_env(cfg)
    return cfg

# ---------- Redis helpers ----------

def init_redis_from_config(cfg: Dict[str, Any]) -> Tuple[Optional[redis.Redis], str]:
    persistence = cfg.get("persistence", {})
    url = persistence.get("redis_url") or os.environ.get("REDIS_URL")
    if not url:
        print("[perf_common] ERROR: No Redis URL (persistence.redis_url or REDIS_URL).")
        return None, persistence.get("key_prefix", "spideybot:v1")

    prefix = persistence.get("key_prefix", "spideybot:v1")
    ttl_minutes = int(persistence.get("ttl_minutes", 2880))

    try:
        r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        r.setex(f"{prefix}:perf_selftest", ttl_minutes * 60, pd.Timestamp.utcnow().isoformat())
        print(f"[perf_common] Redis OK | prefix={prefix}")
        return r, prefix
    except Exception as e:
        print(f"[perf_common] Redis init error: {e}")
        return None, prefix

def redis_load_json(r: redis.Redis, prefix: str, *parts: str, default=None):
    key = ":".join([prefix, *parts])
    if r is None:
        print(f"[perf_common] redis_load_json called with r=None for key {key}")
        return default if default is not None else {}
    try:
        txt = r.get(key)
    except Exception as e:
        print(f"[perf_common] Redis GET error on {key}: {e}")
        return default if default is not None else {}
    if not txt:
        return default if default is not None else {}
    try:
        return json.loads(txt)
    except Exception as e:
        print(f"[perf_common] JSON decode error for {key}: {e}")
        return default if default is not None else {}

# ---------- Trades loader ----------

def load_trades(r: Optional[redis.Redis], prefix: str) -> List[Dict[str, Any]]:
    """
    Load closed Movers trades from the same Redis used by mexc_movers_bot.py
    (state:silent_open, filtered by source=movers & status=closed).
    """
    if r is None:
        print("[perf_common] load_trades: Redis client is None, returning empty list.")
        return []

    book = redis_load_json(r, prefix, "state", "silent_open", default={})
    trades: List[Dict[str, Any]] = []

    for tr in book.values():
        if tr.get("source") != "movers":
            continue
        if tr.get("status") != "closed":
            continue
        trades.append(tr)

    print(f"[perf_common] Loaded {len(trades)} closed mover trades from Redis.")
    return trades

# ---------- Discord helpers ----------

def init_discord_from_config(cfg: Dict[str, Any]) -> str:
    """
    Priority:
      1) DISCORD_PERF_WEBHOOK env (if you want a dedicated performance channel)
      2) cfg.discord.signals_webhook (reuse signals channel)
    """
    hook = os.environ.get("DISCORD_PERF_WEBHOOK") or cfg.get("discord", {}).get("signals_webhook", "") or ""
    if not hook:
        print("[perf_common] WARNING: No Discord webhook configured (DISCORD_PERF_WEBHOOK or discord.signals_webhook).")
    else:
        print("[perf_common] Using Discord webhook from env/config.")
    return hook

def post_discord(hook: str, text: str):
    if not text.strip():
        print("[perf_common] Empty message, not sending to Discord.")
        return
    if not hook:
        print("[perf_common] No webhook; not sending to Discord. Message was:\n", text)
        return
    try:
        resp = requests.post(hook, json={"content": text}, timeout=10)
        print(f"[perf_common] Discord POST status: {resp.status_code}")
    except Exception as e:
        print("[perf_common] Discord post error:", e)
