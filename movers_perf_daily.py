#!/usr/bin/env python3
import argparse
import datetime
import pandas as pd

from perf_common import (
    load_config,
    init_redis_from_config,
    init_discord_from_config,
    load_trades,
    post_discord,
)


def run(cfg):
    # Get performance webhook (DISCORD_PERFORMANCE_WEBHOOK or fallback)
    hook = init_discord_from_config(cfg)

    # 🔹 IMPORTANT: load_trades takes NO arguments
    trades = load_trades()

    if not trades:
        print("[daily] No closed trades found in Redis.")
        post_discord(hook, "📊 No closed trades yet.")
        return

    today = datetime.datetime.utcnow().date()
    today_trades = [
        t for t in trades
        if t.get("closed_at") and pd.to_datetime(t["closed_at"]).date() == today
    ]

    if not today_trades:
        print("[daily] No trades closed today.")
        # Optional: post nothing if empty day
        # post_discord(hook, f"🗓️ Daily Movers Recap — {today.isoformat()}\nNo trades closed today.")
        return

    print(f"[daily] Found {len(today_trades)} trades closed today.")

    n = len(today_trades)
    wins = [t for t in today_trades if t.get("outcome") in ("t1", "t2")]
    stops = [t for t in today_trades if t.get("outcome") == "stop"]
    win_rate = (len(wins) / n) * 100 if n else 0.0

    Rs = [float(t["R"]) for t in today_trades if t.get("R") is not None]
    cum_R = sum(Rs) if Rs else 0.0

    best = max(today_trades, key=lambda t: t.get("R", 0))
    worst = min(today_trades, key=lambda t: t.get("R", 0))

    msg = f"""🗓️ **Daily Movers Recap — {today.isoformat()}**
Closed: **{n}** | Win rate: **{win_rate:.1f}%** | Total: **{cum_R:+.2f}R**
Best: `{best['symbol']}` {best.get('R', 0):+,.2f}R | Worst: `{worst['symbol']}` {worst.get('R', 0):+,.2f}R
━━━━━━━━━━━━━━
Recent:
""" + "\n".join(
        f"{t['symbol']} — {t['outcome'].upper()} — {t.get('R', 0):+,.2f}R — "
        f"{(t.get('time_to_outcome_min', 0) or 0) / 60:.1f}h"
        for t in today_trades[-10:]
    )

    print("[daily] Sending Discord daily recap…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # This sets up Redis inside perf_common (no need to pass r/prefix around)
    init_redis_from_config(cfg)

    run(cfg)
