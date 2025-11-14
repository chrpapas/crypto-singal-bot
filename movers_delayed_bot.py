#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
movers_delayed_bot.py — Delayed preview publisher for Movers signals.

- Reads movers signals from Redis (same as mexc_movers_bot.py)
- After a configurable delay, posts a sanitized "preview" to a
  separate Discord channel (no entry/SL/TP/percentages).
- Marks each signal as 'delayed_preview_sent' to avoid duplicates.

ENV / CONFIG:
  REDIS_URL                       (or persistence.redis_url in config)
  DISCORD_DELAYED_WEBHOOK         (or discord.delayed_webhook in config)
  ALPHA_UPGRADE_URL               (optional; falls back to placeholder)
  DELAY_MINUTES                   (optional; default: 30)
"""

import argparse
import json
import os
import sys
import yaml
import requests
import pandas as pd
import redis
from typing import Dict, Any, Optional

# =============== Config helpers ===============

def load_config(path: str) -> Dict[str, Any]:
    print(f"[delayed] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def expand_env(o):
        if isinstance(o, dict):
            return {k: expand_env(v) for k, v in o.items()}
        if isinstance(o, list):
            return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o

    cfg = expand_env(cfg)
    return cfg

# =============== Redis wrapper ===============

class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url:
            raise RuntimeError("Redis URL missing. Provide REDIS_URL or persistence.redis_url in config.")
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48 * 3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        # self-test
        self.r.setex(self.k("delayed_selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[delayed] Redis OK | prefix={self.prefix} | ttl-min={self.ttl_seconds//60}")

    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])

    def get(self, *parts) -> Optional[str]:
        return self.r.get(self.k(*parts))

    def load_json(self, *parts, default=None):
        txt = self.get(*parts)
        if not txt:
            return default if default is not None else {}
        try:
            return json.loads(txt)
        except Exception:
            return default if default is not None else {}

    def save_json(self, obj, *parts):
        self.r.set(self.k(*parts), json.dumps(obj))

# =============== Discord helper ===============

def get_delayed_webhook(cfg: Dict[str, Any]) -> str:
    hook = os.environ.get("DISCORD_DELAYED_WEBHOOK") or cfg.get("discord", {}).get("delayed_webhook", "")
    if not hook or "${DISCORD_DELAYED_WEBHOOK}" in str(hook):
        print("[delayed] ERROR: No valid DISCORD_DELAYED_WEBHOOK / discord.delayed_webhook configured.", file=sys.stderr)
        return ""
    print("[delayed] Using delayed Discord webhook from env/config.")
    return str(hook)

def post_discord(hook: str, text: str):
    if not hook:
        print("[delayed] No webhook configured; skipping Discord post.", file=sys.stderr)
        return
    if not text.strip():
        return
    try:
        r = requests.post(hook, json={"content": text}, timeout=10)
        if r.status_code >= 300:
            print(f"[delayed] Discord post failed: HTTP {r.status_code} {r.text}", file=sys.stderr)
        else:
            print("[delayed] Discord post OK.")
    except Exception as e:
        print(f"[delayed] Discord post error: {e}", file=sys.stderr)

# =============== Message formatting ===============

def fmt_delayed_preview(tr: Dict[str, Any], delay_minutes: int, upgrade_url: str) -> str:
    symbol = tr.get("symbol", "UNKNOWN")
    tf = tr.get("tf") or tr.get("timeframe") or "1h"
    typ = (tr.get("type") or "").lower()
    note = tr.get("note") or "Mover Signal"
    opened_at = tr.get("opened_at")
    opened_ts = pd.to_datetime(opened_at, utc=True) if opened_at else None
    now = pd.Timestamp.utcnow().tz_localize("UTC") if not pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow()
    age_min = None
    if opened_ts is not None:
        age_min = (now - opened_ts).total_seconds() / 60.0

    # Human text for delay (rounded actual age)
    delay_txt = f"{int(round(age_min))} minutes" if age_min is not None else f"{delay_minutes} minutes"

    # Generic readable type
    if typ == "day":
        typ_label = "1h Breakout"
    else:
        typ_label = "Mover Breakout"

    header = "**Kriticurrency Alpha Signals – Movers Signal (Delayed Preview) 🚀**"

    lines = [
        header,
        "",
        f"**Symbol:** `{symbol}`",
        f"**Timeframe:** {tf}",
        f"**Pattern:** {typ_label}",
        f"**Engine:** AI-powered Movers breakout scanner",
        "",
        f"⏱️ *This preview is delayed by about* **{delay_txt}**.",
        f"Realtime entries, stop-loss, targets & risk ratings are available only to **Alpha Members**.",
        "",
        f"🔓 Upgrade to realtime access → {upgrade_url}",
    ]

    msg = "\n".join(lines)
    # Just in case, hard-limit to 2000 chars for Discord
    if len(msg) > 2000:
        msg = msg[:1990] + "…"
    return msg

# =============== Core logic ===============

def run(cfg: Dict[str, Any]):
    print("[delayed] Starting delayed movers preview bot…")

    # Delay config
    delay_minutes = int(os.environ.get("DELAY_MINUTES", "30"))
    print(f"[delayed] Using delay threshold: {delay_minutes} minutes")

    upgrade_url = os.environ.get("ALPHA_UPGRADE_URL", "https://your-landing-page-here")
    print(f"[delayed] Using upgrade URL: {upgrade_url}")

    # Redis init
    redis_url = cfg.get("persistence", {}).get("redis_url") or os.environ.get("REDIS_URL")
    prefix = cfg.get("persistence", {}).get("key_prefix", "spideybot:v1")
    ttl = int(cfg.get("persistence", {}).get("ttl_minutes", 2880))
    rds = RedisState(redis_url, prefix, ttl)

    # Discord webhook
    hook = get_delayed_webhook(cfg)
    if not hook:
        print("[delayed] Exiting because no delayed webhook is configured.", file=sys.stderr)
        return

    # Load trade book
    book = rds.load_json("state", "silent_open", default={})
    if not book:
        print("[delayed] No trades found in Redis. Nothing to preview.")
        return

    now = pd.Timestamp.utcnow().tz_localize("UTC") if not pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow()
    delay_seconds = delay_minutes * 60

    # Iterate over movers trades and find those eligible for delayed preview
    to_mark = []
    num_candidates = 0
    num_posted = 0

    for k, tr in book.items():
        # Only movers source trades (created by movers bot)
        if tr.get("source") != "movers":
            continue

        # Only signals that are still open (ignore already closed trades for preview)
        if tr.get("status") != "open":
            continue

        # Skip if preview already sent
        if tr.get("delayed_preview_sent"):
            continue

        opened_at = tr.get("opened_at")
        if not opened_at:
            continue

        try:
            opened_ts = pd.to_datetime(opened_at, utc=True)
        except Exception:
            continue

        age_sec = (now - opened_ts).total_seconds()
        if age_sec < delay_seconds:
            # Not yet past delay threshold
            continue

        num_candidates += 1

        # Build and send preview
        msg = fmt_delayed_preview(tr, delay_minutes, upgrade_url)
        print(f"[delayed] Posting delayed preview for {tr.get('symbol')} (opened_at={opened_at})")
        post_discord(hook, msg)

        # Mark to update
        tr["delayed_preview_sent"] = True
        tr["delayed_preview_sent_at"] = now.isoformat()
        book[k] = tr
        num_posted += 1

    if num_posted > 0:
        rds.save_json(book, "state", "silent_open")
        print(f"[delayed] Marked {num_posted} trades as delayed_preview_sent.")
    else:
        print("[delayed] No new signals eligible for delayed preview right now.")

    print(f"[delayed] Done. Candidates checked: {num_candidates}, posted: {num_posted}")

# =============== Entrypoint ===============

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
