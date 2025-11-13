#!/usr/bin/env python3
import argparse, datetime
import pandas as pd

from perf_common import (
    load_config,
    init_redis_from_config,
    init_discord_from_config,
    load_trades,
    post_discord,
)


def run(cfg):
    hook = init_discord_from_config(cfg)
    trades = load_trades()

    if not trades:
        post_discord(hook, "📊 No closed trades yet.")
        return

    today = datetime.datetime.utcnow().date()
    cutoff = today - datetime.timedelta(days=30)

    recent = []
    for t in trades:
        ca = t.get("closed_at")
        if not ca:
            continue
        d = pd.to_datetime(ca).date()
        if d >= cutoff:
            recent.append(t)

    if not recent:
        print("[spotlight] No trades in last 30 days.")
        return

    # Pick the biggest winner by R
    winners = [t for t in recent if (t.get("R") or 0.0) > 0]
    if not winners:
        print("[spotlight] No winning trades; picking least bad trade.")
        spotlight = max(recent, key=lambda t: (t.get("R") or 0.0))
    else:
        spotlight = max(winners, key=lambda t: float(t.get("R", 0.0)))

    sym = spotlight.get("symbol", "UNKNOWN")
    outcome = (spotlight.get("outcome") or "").upper()
    R = spotlight.get("R")
    entry = spotlight.get("entry")
    exit_price = spotlight.get("exit_price")
    opened_at = spotlight.get("opened_at")
    closed_at = spotlight.get("closed_at")
    tmin = spotlight.get("time_to_outcome_min", 0.0) or 0.0
    hours = tmin / 60.0

    msg = (
        f"✨ **Movers Trade Spotlight — last 30 days**\n"
        f"Highlighting one of the top trades from the last month.\n"
        f"━━━━━━━━━━━━━━\n"
        f"**Pair:** `{sym}`\n"
        f"**Outcome:** {outcome}\n"
    )

    if R is not None:
        msg += f"**Result:** {float(R):+0.2f}R\n"
    if entry is not None and exit_price is not None:
        msg += f"**Entry:** `{float(entry):.6f}`  →  **Exit:** `{float(exit_price):.6f}`\n"
    if opened_at and closed_at:
        msg += f"**Opened:** {opened_at}\n**Closed:** {closed_at}\n"
    msg += f"**Time in trade:** ~{hours:.1f} hours\n"

    msg += "\n_This is an example of what a strong Movers breakout can look like. Not financial advice._"

    print("[spotlight] Sending Discord spotlight…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_redis_from_config(cfg)
    run(cfg)
