# perf_common.py
import os, json, time, math
import redis
import pandas as pd
import requests

# ----- Discord -----
DISCORD_WEBHOOK = os.getenv("DISCORD_PERF_WEBHOOK") or os.getenv("DISCORD_SIGNALS_WEBHOOK", "")

def post_discord(text: str):
    if not DISCORD_WEBHOOK or not text:
        print("[perf] Discord webhook missing or empty message; skipping post.")
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": text}, timeout=10)
        print("[perf] Posted to Discord.")
    except Exception as e:
        print("[perf] Discord post error:", e)

# ----- Redis client (Upstash-safe) -----
_REDIS = None

def _make_redis():
    url = os.getenv("REDIS_URL")  # keep creds in env on Render
    if not url:
        raise RuntimeError("REDIS_URL is not set in environment.")
    # Upstash over TLS uses rediss://. Render egress ok.
    # Tight timeouts + retry_on_timeout + keepalive.
    return redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=5,   # connect timeout
        socket_timeout=5,          # read/write timeout
        retry_on_timeout=True,
        health_check_interval=30,  # PING occasionally
    )

def _get_redis():
    global _REDIS
    if _REDIS is None:
        _REDIS = _make_redis()
    return _REDIS

def redis_load_json(*key_parts, default=None, retries=4):
    """
    Resilient getter with exponential backoff. Returns `default` on failure.
    """
    key = ":".join([str(k) for k in key_parts])
    delay = 1.0
    for attempt in range(retries):
        try:
            r = _get_redis()
            txt = r.get(key)
            if not txt:
                return default if default is not None else {}
            return json.loads(txt)
        except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError) as e:
            print(f"[perf] Redis timeout/conn error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
            delay *= 2
            # force reconnect next try
            global _REDIS
            _REDIS = None
        except Exception as e:
            print("[perf] Redis get error:", e)
            break
    print("[perf] Giving up Redis read; returning default.")
    return default if default is not None else {}

def load_trades():
    """
    Returns list of CLOSED movers trades from silent registry.
    """
    book = redis_load_json("spideybot:v1", "state", "silent_open", default={})
    if not isinstance(book, dict):
        print("[perf] Invalid book shape; treating as no data.")
        return []
    trades = [tr for tr in book.values() if tr.get("source") == "movers" and tr.get("status") == "closed"]
    # Normalize a few fields
    for t in trades:
        # Ensure numerics are castable
        for k in ("entry","stop","t1","t2","exit_price","R","time_to_outcome_min","time_to_t1_min","time_to_t2_min"):
            if k in t and t[k] is not None:
                try:
                    t[k] = float(t[k])
                except Exception:
                    t[k] = None
    return trades

# ----- Common format helpers -----
def fmt_R(val):
    try:
        return f"{float(val):+.2f}R"
    except Exception:
        return "n/a"

def safe_hours(mins):
    try:
        return float(mins) / 60.0 if mins is not None else 0.0
    except Exception:
        return 0.0
