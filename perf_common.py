import os, yaml, json, redis, requests, pandas as pd

def load_config(path: str):
    print(f"[perf_common] Loading config from {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def expand_env(o):
        if isinstance(o, dict):
            return {k: expand_env(v) for k, v in o.items()}
        if isinstance(o, list):
            return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            env_key = o[2:-1]
            return os.environ.get(env_key, o)
        return o

    cfg = expand_env(raw)
    return cfg


def init_redis_from_config(cfg):
    from redis import Redis
    p = cfg.get("persistence", {})
    url = os.environ.get("REDIS_URL") or p.get("redis_url")
    prefix = p.get("key_prefix", "spideybot:v1")
    ttl_minutes = int(p.get("ttl_minutes", 2880))

    if not url:
        raise RuntimeError("No REDIS_URL or persistence.redis_url configured")

    r = Redis.from_url(url, decode_responses=True, socket_timeout=6)
    r.setex(f"{prefix}:perf_selftest", ttl_minutes * 60, "ok")
    print(f"[perf_common] Redis OK | prefix={prefix}")
    return r, prefix


def init_discord_from_config(cfg):
    dcfg = cfg.get("discord", {})

    # 1) Prefer dedicated performance env var
    hook = os.environ.get("DISCORD_PERFORMANCE_WEBHOOK")

    # 2) Then dedicated performance key in config
    if not hook:
        hook = dcfg.get("performance_webhook", "")

    # 3) Fallback to signals webhook env / config if needed
    if not hook:
        hook = os.environ.get("DISCORD_SIGNALS_WEBHOOK") or dcfg.get("signals_webhook", "")

    # If it's still a ${...} placeholder, treat as missing
    if hook and hook.startswith("${") and hook.endswith("}"):
        print("[perf_common] Discord webhook still looks like a placeholder, not a real URL. Skipping posts.")
        hook = ""

    if hook:
        print("[perf_common] Using Discord webhook from env/config.")
    else:
        print("[perf_common] No Discord webhook configured; will not post.")

    return hook


def redis_load_json(*parts, default=None):
    global _redis_client, _redis_prefix
    key = f"{_redis_prefix}:" + ":".join(str(p) for p in parts)
    txt = _redis_client.get(key)
    if not txt:
        return default if default is not None else {}
    try:
        return json.loads(txt)
    except Exception:
        return default if default is not None else {}


def load_trades():
    book = redis_load_json("state", "silent_open", default={})
    movers = [
        tr for tr in book.values()
        if tr.get("source") == "movers" and tr.get("status") == "closed"
    ]
    print(f"[perf_common] Loaded {len(movers)} closed mover trades from Redis.")
    return movers


def post_discord(hook: str, text: str):
    if not hook:
        print("[perf_common] No webhook URL; skipping Discord post.")
        return
    try:
        requests.post(hook, json={"content": text}, timeout=10)
    except Exception as e:
        print(f"[perf_common] Discord post error: {e}")


# globals used by redis_load_json
_redis_client = None
_redis_prefix = None

def set_redis_globals(r, prefix):
    global _redis_client, _redis_prefix
    _redis_client = r
    _redis_prefix = prefix
