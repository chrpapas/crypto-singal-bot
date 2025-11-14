#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
movers_backfill_timing.py

Backfill time_to_t1_min and time_to_t2_min for existing closed Movers trades
stored in Redis under:  <prefix>:state:silent_open

- Uses the same config file as your movers bot (e.g. movers-signals-config.yml)
- Does NOT change outcome or exit_price
- Only updates:
    - time_to_t1_min
    - time_to_t2_min
"""

import argparse
import json
import os
import yaml
import datetime

from typing import Any, Dict

import pandas as pd
import ccxt
import redis


# =============== Config loader (same style as your bots) ===============
def expand_env(o):
    if isinstance(o, dict):
        return {k: expand_env(v) for k, v in o.items()}
    if isinstance(o, list):
        return [expand_env(x) for x in o]
    if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
        return os.environ.get(o[2:-1], o)
    return o


def load_config(path: str) -> Dict[str, Any]:
    print(f"[backfill] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = expand_env(cfg)
    return cfg


# =============== Redis helpers ===============
def init_redis_from_config(cfg: Dict[str, Any]) -> redis.Redis:
    p = cfg.get("persistence", {}) or {}
    url = p.get("redis_url") or os.environ.get("REDIS_URL")
    prefix = p.get("key_prefix", "spideybot:v1")

    if not url:
        raise RuntimeError("Redis URL missing in config.persistence.redis_url or REDIS_URL")

    r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=8)
    # quick self-test
    test_key = f"{prefix}:backfill_selftest"
    r.setex(test_key, 60, datetime.datetime.utcnow().isoformat())
    print(f"[backfill] Redis OK | prefix={prefix}")
    return r


def redis_load_json(r: redis.Redis, prefix: str, *parts, default=None):
    key = ":".join([prefix, *[str(p) for p in parts]])
    txt = r.get(key)
    if not txt:
        return default if default is not None else {}
    try:
        return json.loads(txt)
    except Exception:
        return default if default is not None else {}


def redis_save_json(r: redis.Redis, prefix: str, obj, *parts):
    key = ":".join([prefix, *[str(p) for p in parts]])
    r.set(key, json.dumps(obj))


# =============== Exchange wrapper (minimal) ===============
class ExClient:
    def __init__(self):
        self.ex = ccxt.mexc({"enableRateLimit": True})
        key = os.environ.get("MEXC_API_KEY")
        sec = os.environ.get("MEXC_SECRET")
        if key and sec:
            self.ex.apiKey = key
            self.ex.secret = sec

    def ohlcv(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")


# =============== Backfill logic ===============
def backfill_timing(cfg: Dict[str, Any]):
    # Init Redis
    p = cfg.get("persistence", {}) or {}
    prefix = p.get("key_prefix", "spideybot:v1")
    r = init_redis_from_config(cfg)

    # Init exchange
    client = ExClient()

    # Load book
    book = redis_load_json(r, prefix, "state", "silent_open", default={})
    if not book:
        print("[backfill] No trades found in Redis (silent_open is empty).")
        return

    # Filter only closed movers trades
    trades = [
        (k, tr) for k, tr in book.items()
        if tr.get("source") == "movers" and tr.get("status") == "closed"
    ]

    print(f"[backfill] Found {len(trades)} closed movers trades to inspect.")

    updated = 0
    skipped = 0
    errors = 0

    for key, tr in trades:
        symbol = tr.get("symbol")
        tf = tr.get("tf") or tr.get("timeframe", "1h")
        opened_at = tr.get("opened_at")
        t1 = tr.get("t1")
        t2 = tr.get("t2")

        if not symbol or not opened_at:
            skipped += 1
            continue

        # If both timing fields already exist and are non-null, you can skip,
        # or comment out this block if you want to recompute everything.
        if tr.get("time_to_t1_min") is not None or tr.get("time_to_t2_min") is not None:
            # You can choose to skip or recompute; here we recompute only missing.
            pass

        try:
            opened_ts = pd.to_datetime(opened_at, utc=True)

            limit = 400 if tf in ("1h", "4h") else 330
            df = client.ohlcv(symbol, tf, limit)

            # Focus from signal bar onwards
            df = df.loc[opened_ts:]
            if len(df) <= 1:
                skipped += 1
                continue

            rows = df.iloc[1:]  # bars after signal bar

            t1_ts = None
            t2_ts = None

            for ts, row in rows.iterrows():
                hi = float(row["high"])

                if t1 is not None and t1_ts is None and hi >= float(t1):
                    t1_ts = ts
                if t2 is not None and t2_ts is None and hi >= float(t2):
                    t2_ts = ts

                # If both found, we can stop early
                if t1_ts is not None and t2_ts is not None:
                    break

            # Fill timing fields (minutes from opened_at)
            changed = False

            if t1_ts is not None:
                dt1 = (t1_ts - opened_ts).total_seconds() / 60.0
                tr["time_to_t1_min"] = float(dt1)
                changed = True

            if t2_ts is not None:
                dt2 = (t2_ts - opened_ts).total_seconds() / 60.0
                tr["time_to_t2_min"] = float(dt2)
                changed = True

            if changed:
                book[key] = tr
                updated += 1
            else:
                skipped += 1

        except Exception as e:
            errors += 1
            print(f"[backfill] Error for {symbol} {tf}: {e}")

    # Save back to Redis
    if updated > 0:
        redis_save_json(r, prefix, book, "state", "silent_open")

    print("----------- Backfill summary -----------")
    print(f"Closed movers trades inspected: {len(trades)}")
    print(f"Updated timing fields         : {updated}")
    print(f"Skipped (no change)           : {skipped}")
    print(f"Errors                        : {errors}")
    print("----------------------------------------")


# =============== Entrypoint ===============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backfill_timing(cfg)
