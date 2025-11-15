#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
movers_delayed_signals_bot.py — Delayed-preview bot for Movers signals

- Reads Movers signals from the same Redis used by mexc_movers_bot.py
- Publishes delayed previews to a separate Discord channel
- Optionally posts short delayed previews to Twitter/X
- Uses config file (same style as movers-signals-config.yml)

ENV (optional, only for Discord if you still want):
  REDIS_URL
  DISCORD_DELAYED_WEBHOOK

Config (YAML example):

  persistence:
    redis_url: "rediss://..."
    key_prefix: spideybot:v1
    ttl_minutes: 2880

  discord:
    # Main realtime signals webhook (not used here, but kept for compatibility)
    signals_webhook: "${DISCORD_SIGNALS_WEBHOOK}"

  delayed_signals:
    webhook: "https://discord.com/api/webhooks/...."  # delayed channel webhook
    min_delay_minutes: 240         # how long after signal before preview allowed
    max_age_days: 14               # ignore signals older than this
    max_previews_per_run: 5        # cap per run to avoid spam

  twitter:
    enabled: true
    bearer_token: "YOUR_USER_CONTEXT_BEARER_TOKEN_HERE"
    promo_url: "https://discord.gg/U2g7qHZP"   # optional: link to your Discord/landing
"""

import argparse
import json
import os
import sys
from datetime import timedelta

import pandas as pd
import redis
import requests
import yaml

# ---------------- Config helpers ----------------


def expand_env(o):
    if isinstance(o, dict):
        return {k: expand_env(v) for k, v in o.items()}
    if isinstance(o, list):
        return [expand_env(x) for x in o]
    if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
        # Allow env expansion for backwards compat, but config
        # can also contain plain values with no env usage.
        return os.environ.get(o[2:-1], o)
    return o


def load_config(path: str) -> dict:
    print(f"[delayed] Loading config from {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = expand_env(cfg)
    return cfg


# ---------------- Redis helpers ----------------

def init_redis_from_config(cfg: dict):
    """
    Returns (redis_client, key_prefix)
    """
    p = cfg.get("persistence", {}) or {}
    redis_url = p.get("redis_url") or os.environ.get("REDIS_URL")
    if not redis_url:
        print("[delayed] ERROR: redis_url not set in config and REDIS_URL env missing.", file=sys.stderr)
        sys.exit(1)

    key_prefix = p.get("key_prefix", "spideybot:v1")
    ttl_minutes = int(p.get("ttl_minutes", 2880))

    r = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=6)
    # self-test + keep TTL behavior similar to main bot
    r.setex(f"{key_prefix}:selftest_delayed", ttl_minutes * 60, pd.Timestamp.utcnow().isoformat())

    print(f"[delayed] Redis OK | prefix={key_prefix}")
    return r, key_prefix


def redis_key(prefix: str, *parts) -> str:
    return ":".join([prefix, *[str(p) for p in parts]])


def redis_load_json(r: redis.Redis, prefix: str, *parts, default=None):
    key = redis_key(prefix, *parts)
    txt = r.get(key)
    if not txt:
        return default if default is not None else {}
    try:
        return json.loads(txt)
    except Exception:
        return default if default is not None else {}


def redis_save_json(r: redis.Redis, prefix: str, obj, *parts):
    key = redis_key(prefix, *parts)
    r.set(key, json.dumps(obj))


# ---------------- Discord helper ----------------

def pick_delayed_webhook(cfg: dict) -> str:
    dcfg = cfg.get("delayed_signals", {}) or {}
    discord_cfg = cfg.get("discord", {}) or {}

    # Priority:
    # 1) delayed_signals.webhook in config (after env expansion)
    # 2) DISCORD_DELAYED_WEBHOOK env
    # 3) discord.signals_webhook (main signals) as last fallback
    hook = dcfg.get("webhook") or os.environ.get("DISCORD_DELAYED_WEBHOOK") or discord_cfg.get("signals_webhook", "")

    if not hook or str(hook).startswith("${"):
        print("[delayed] WARNING: No valid delayed Discord webhook configured.")
        return ""
    print("[delayed] Using Discord delayed webhook from env/config.")
    return hook


def post_discord(hook: str, text: str):
    if not hook:
        print("[delayed] No webhook provided; skipping Discord post.")
        return
    if not text.strip():
        print("[delayed] Empty message; skipping Discord post.")
        return
    try:
        resp = requests.post(hook, json={"content": text}, timeout=10)
        if resp.status_code >= 300:
            print(f"[delayed] Discord post failed: HTTP {resp.status_code} {resp.text}")
        else:
            print("[delayed] Discord post OK.")
    except Exception as e:
        print(f"[delayed] Discord post error: {e}")


# ---------------- Twitter helpers ----------------

def init_twitter_from_config(cfg: dict):
    """
    Returns (bearer_token, promo_url) or (None, None) if disabled/misconfigured.

    bearer_token should be a *user-context* OAuth2 token that can post tweets.
    """
    tw = cfg.get("twitter", {}) or {}
    enabled = bool(tw.get("enabled", False))
    if not enabled:
        print("[delayed] Twitter disabled in config.")
        return None, None

    bearer = tw.get("bearer_token")
    if not bearer or str(bearer).startswith("${"):
        print("[delayed] Twitter enabled but bearer_token missing or placeholder; skipping Twitter.")
        return None, None

    promo_url = tw.get("promo_url") or ""
    print("[delayed] Twitter posting enabled via config.")
    return bearer, promo_url


def post_twitter(bearer_token: str, text: str):
    if not bearer_token:
        return
    if not text.strip():
        print("[delayed] Empty tweet text; skipping Twitter post.")
        return

    # Hard trim to 280 characters
    text = text.strip()
    if len(text) > 280:
        text = text[:277] + "…"

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    try:
        resp = requests.post(
            "https://api.twitter.com/2/tweets",
            headers=headers,
            json=payload,
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            print(f"[delayed] Twitter post failed: HTTP {resp.status_code} {resp.text}")
        else:
            print("[delayed] Twitter post OK.")
    except Exception as e:
        print(f"[delayed] Twitter post error: {e}")


# ---------------- Formatting helpers ----------------

def fmt_delayed_discord_message(tr: dict, delay_minutes: float) -> str:
    """
    Build the delayed preview Discord message for a single mover trade.
    No upsell line – you keep upsell in channel pinned message.
    """
    symbol = tr.get("symbol", "?")
    tf = tr.get("tf") or tr.get("timeframe") or "1h"
    note = tr.get("note") or "Mover Trend"
    opened_at = tr.get("opened_at") or tr.get("bar_id") or "unknown"

    delay_min_int = int(delay_minutes)
    delay_hours = delay_minutes / 60.0

    msg = (
        "**Kritocurrency Alpha Signals – Movers Signal (Delayed Preview) 🚀**\n\n"
        f"**Symbol:** `{symbol}`\n"
        f"**Timeframe:** {tf}\n"
        f"**Pattern:** {note}\n"
        f"**Engine:** AI-powered Movers breakout scanner\n\n"
        f"⏱️ *Original signal time:* `{opened_at}` (UTC)\n"
        f"⏱️ *This preview is delayed by about* **{delay_min_int} minutes** "
        f"(~{delay_hours:.1f} hours).\n\n"
        "Realtime entries, stop-loss, targets & risk ratings are available only to **Alpha Members**."
    )
    return msg


def fmt_delayed_tweet(tr: dict, delay_minutes: float, promo_url: str = "") -> str:
    """
    Short tweet-format version of the delayed preview.
    Kept concise; trimmed to 280 chars in post_twitter().
    """
    symbol = tr.get("symbol", "?")
    tf = tr.get("tf") or tr.get("timeframe") or "1h"
    note = tr.get("note") or "Mover Trend"
    opened_at = tr.get("opened_at") or tr.get("bar_id") or "unknown"

    delay_hours = delay_minutes / 60.0

    base = (
        f"Delayed Movers signal (preview) 🚀\n"
        f"{symbol} · {tf} · {note}\n"
        f"Original (UTC): {opened_at}\n"
        f"Delay: ~{delay_hours:.1f}h\n"
        f"Realtime entries & risk levels only for Alpha Members on Discord."
    )

    if promo_url:
        base += f"\nJoin: {promo_url}"

    # Add a couple of hashtags for reach
    base += "\n#crypto #trading #altcoins"

    return base


# ---------------- Core logic ----------------

def run(cfg: dict):
    print("[delayed] Starting delayed Movers preview run...")

    # Redis + Discord + Twitter
    r, prefix = init_redis_from_config(cfg)
    hook = pick_delayed_webhook(cfg)
    bearer_token, promo_url = init_twitter_from_config(cfg)

    delayed_cfg = cfg.get("delayed_signals", {}) or {}
    min_delay_minutes = int(delayed_cfg.get("min_delay_minutes", 240))
    max_age_days = int(delayed_cfg.get("max_age_days", 14))
    max_previews_per_run = int(delayed_cfg.get("max_previews_per_run", 5))

    print(
        f"[delayed] Settings: min_delay={min_delay_minutes}min, "
        f"max_age={max_age_days}d, cap_per_run={max_previews_per_run}"
    )

    # Load signal book and sent registry
    book = redis_load_json(r, prefix, "state", "silent_open", default={})
    print(f"[delayed] Loaded {len(book)} total signals from Redis.")

    sent_reg = redis_load_json(r, prefix, "state", "delayed_previews_sent", default={})
    print(f"[delayed] Loaded {len(sent_reg)} already-previewed entries.")

    now = pd.Timestamp.utcnow()
    min_delay = timedelta(minutes=min_delay_minutes)
    max_age = timedelta(days=max_age_days)

    # Collect eligible movers signals
    candidates = []
    for k, tr in book.items():
        # Only Movers signals
        if tr.get("source") != "movers":
            continue

        # Already previewed before? Skip.
        if k in sent_reg:
            continue

        opened_raw = tr.get("opened_at") or tr.get("bar_id")
        if not opened_raw:
            continue

        try:
            opened_at = pd.to_datetime(opened_raw, utc=True)
        except Exception:
            continue

        age = now - opened_at

        # Too fresh -> not yet eligible
        if age < min_delay:
            continue

        # Too old -> ignore (but do NOT mark as sent; keeps behavior simple if you change config later)
        if age > max_age:
            continue

        candidates.append((k, tr, opened_at, age))

    print(f"[delayed] Found {len(candidates)} eligible delayed movers signals.")

    if not candidates:
        print("[delayed] Nothing to send this run.")
        return

    # Oldest first, so backlog drains in chronological order
    candidates.sort(key=lambda x: x[2])

    posted = 0
    for k, tr, opened_at, age in candidates:
        if posted >= max_previews_per_run:
            break

        delay_minutes = age.total_seconds() / 60.0

        # Discord message
        d_msg = fmt_delayed_discord_message(tr, delay_minutes)
        post_discord(hook, d_msg)

        # Twitter message (if configured)
        if bearer_token:
            t_msg = fmt_delayed_tweet(tr, delay_minutes, promo_url)
            post_twitter(bearer_token, t_msg)

        # Mark as previewed
        sent_reg[k] = now.isoformat()
        posted += 1

    # Save updated registry
    redis_save_json(r, prefix, sent_reg, "state", "delayed_previews_sent")
    print(f"[delayed] Posted {posted} delayed previews this run.")


# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
