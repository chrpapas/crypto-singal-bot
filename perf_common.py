import os, json, redis, pandas as pd, numpy as np, requests

REDIS_URL = os.environ.get("REDIS_URL")
REDIS_PREFIX = "spideybot:v1"
DISCORD_HOOK = os.environ.get("DISCORD_PERFORMANCE_WEBHOOK")

def redis_load_json(*parts, default=None):
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    key = ":".join([REDIS_PREFIX, *parts])
    txt = r.get(key)
    if not txt: return default or {}
    try:
        return json.loads(txt)
    except Exception:
        return default or {}

def load_trades(source="movers"):
    book = redis_load_json("state", "silent_open", default={})
    return [
        tr for tr in book.values()
        if tr.get("source") == source and tr.get("status") == "closed"
    ]

def post_discord(text: str):
    if DISCORD_HOOK:
        try:
            requests.post(DISCORD_HOOK, json={"content": text}, timeout=10)
        except Exception as e:
            print("[discord] err:", e)
