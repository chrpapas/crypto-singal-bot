#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backfill_movers_timings.py

One-off maintenance script to:
- Backfill time_to_outcome_min for closed Movers trades using opened_at / closed_at
- Backfill R for closed Movers trades if missing

Uses the same Redis + key_prefix as your signals config.
Run once, e.g.:

  python3 backfill_movers_timings.py --config movers-signals-config.yml
"""

import argparse
import json
import os

import pandas as pd
import redis
import yaml


def expand_env(o):
    if isinstance(o, dict):
        return {k: expand_env(v) for k, v in o.items()}
    if isinstance(o, list):
        return [expand_env(x) for x in o]
    if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
        return os.environ.get(o[2:-1], o)
    return o


def load_config(path: str):
    print(f"[backfill] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return expand_env(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Redis settings (same as your bot / perf scripts)
    redis_url = (
        cfg.get("persistence", {}).get("redis_url")
        or os.environ.get("REDIS_URL")
    )
    prefix = cfg.get("persistence", {}).get("key_prefix", "spideybot:v1")

    if not redis_url:
        raise RuntimeError(
            "[backfill] No Redis URL found in config.persistence.redis_url or REDIS_URL"
        )

    print(f"[backfill] Connecting to Redis with prefix={prefix!r} …")
    r = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=8)

    key = f"{prefix}:state:silent_open"
    print(f"[backfill] Loading trades from key: {key}")
    txt = r.get(key)
    if not txt:
        print("[backfill] No data found at that key. Nothing to do.")
        return

    try:
        book = json.loads(txt)
    except Exception as e:
        print("[backfill] ERROR: could not decode JSON from Redis:", e)
        return

    if not isinstance(book, dict):
        print("[backfill] ERROR: unexpected JSON type (expected dict).")
        return

    updated_time = 0
    updated_R = 0
    total_closed_movers = 0

    for k, tr in book.items():
        if not isinstance(tr, dict):
            continue

        if tr.get("source") != "movers":
            continue
        if tr.get("status") != "closed":
            continue

        total_closed_movers += 1

        # ---------- backfill time_to_outcome_min ----------
        cur_tto = tr.get("time_to_outcome_min")
        if cur_tto is None:
            opened_at = tr.get("opened_at")
            closed_at = tr.get("closed_at")
            if opened_at and closed_at:
                try:
                    oa = pd.to_datetime(opened_at, utc=True)
                    ca = pd.to_datetime(closed_at, utc=True)
                    dt_min = (ca - oa).total_seconds() / 60.0
                    if dt_min >= 0:
                        tr["time_to_outcome_min"] = float(dt_min)
                        updated_time += 1
                except Exception as e:
                    print(f"[backfill] time_to_outcome_min parse error for {k}:", e)

        # ---------- backfill R (risk multiple) ----------
        if tr.get("R") is None:
            try:
                entry = float(tr.get("entry"))
                stop_val = tr.get("stop")
                exit_price = tr.get("exit_price")
                if stop_val is not None and exit_price is not None:
                    stop_val = float(stop_val)
                    exit_price = float(exit_price)
                    risk = max(1e-9, entry - stop_val)
                    R_val = (exit_price - entry) / risk
                    tr["R"] = float(R_val)
                    updated_R += 1
            except Exception as e:
                print(f"[backfill] R compute error for {k}:", e)

        # write back updated record
        book[k] = tr

    # Save updated JSON back to Redis
    r.set(key, json.dumps(book))

    print("========== Backfill summary ==========")
    print(f"Total closed movers trades seen : {total_closed_movers}")
    print(f"Updated time_to_outcome_min     : {updated_time}")
    print(f"Updated R                       : {updated_R}")
    print("======================================")
    print("[backfill] Done.")


if __name__ == "__main__":
    main()
