#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    print("[weekly] Starting weekly movers performance recap...")

    # Discord + Redis init (Redis already done in main, but harmless if repeated)
    hook = init_discord_from_config(cfg)

    trades = load_trades()
    if not trades:
        print("[weekly] No closed trades found at all. Skipping.")
        post_discord(hook, "📅 Weekly recap: no closed trades yet.")
        return

    today = datetime.datetime.utcnow().date()
    week_start = today - datetime.timedelta(days=7)

    weekly_trades = [
        t
        for t in trades
        if t.get("closed_at")
        and week_start <= pd.to_datetime(t["closed_at"]).date() <= today
    ]

    if not weekly_trades:
        print("[weekly] No trades closed in the last 7 days.")
        post_discord(
            hook,
            f"📅 **Weekly Movers Recap — {week_start.isoformat()} → {today.isoformat()}**\n"
            "No trades were closed this week."
        )
        return

    print(f"[weekly] Found {len(weekly_trades)} trades closed this week.")

    # Lifetime stats (all closed movers)
    total_closed = len(trades)
    print(f"[weekly] Lifetime closed trades: {total_closed}")

    weekly_wins = [t for t in weekly_trades if t.get("outcome") in ("t1", "t2")]
    weekly_stops = [t for t in weekly_trades if t.get("outcome") == "stop"]

    weekly_win_rate = (len(weekly_wins) / len(weekly_trades)) * 100 if weekly_trades else 0.0

    weekly_Rs = [float(t["R"]) for t in weekly_trades if t.get("R") is not None]
    weekly_cum_R = sum(weekly_Rs) if weekly_Rs else 0.0

    lifetime_wins = [t for t in trades if t.get("outcome") in ("t1", "t2")]
    lifetime_Rs = [float(t["R"]) for t in trades if t.get("R") is not None]
    lifetime_win_rate = (len(lifetime_wins) / total_closed) * 100 if total_closed else 0.0
    lifetime_cum_R = sum(lifetime_Rs) if lifetime_Rs else 0.0

    # Best / worst this week (by R)
    best_week = max(
        weekly_trades,
        key=lambda t: (t.get("R") if t.get("R") is not None else -9999),
    )
    worst_week = min(
        weekly_trades,
        key=lambda t: (t.get("R") if t.get("R") is not None else 9999),
    )

    # ---- Build a compact, line-based message ----
    header = (
        f"📅 **Weekly Movers Recap — {week_start.isoformat()} → {today.isoformat()}**\n"
        f"Closed: **{len(weekly_trades)}** | Wins: **{len(weekly_wins)}** | Stops: **{len(weekly_stops)}**\n"
        f"Week PnL: **{weekly_cum_R:+.2f}R** | Win rate: **{weekly_win_rate:.1f}%**\n"
        f"Best: `{best_week['symbol']}` {best_week.get('R', 0):+.2f}R | "
        f"Worst: `{worst_week['symbol']}` {worst_week.get('R', 0):+.2f}R\n"
        "━━━━━━━━━━━━━━\n"
        f"**Lifetime (all closed Movers trades)**\n"
        f"Total closed: **{total_closed}** | Win rate: **{lifetime_win_rate:.1f}%** | "
        f"Cumulative: **{lifetime_cum_R:+.2f}R**\n"
        "━━━━━━━━━━━━━━\n"
        "**All trades closed this week:**"
    )

    # Per-trade lines — short & consistent so chunking in perf_common works well
    lines = []
    for t in weekly_trades:
        sym = t.get("symbol", "?")
        outcome = (t.get("outcome") or "").upper()
        R = t.get("R")
        R_txt = f"{float(R):+.2f}R" if R is not None else "n/a"
        mins = float(t.get("time_to_outcome_min", 0) or 0)
        hours = mins / 60.0
        closed_at = t.get("closed_at", "")[:16]  # trim for brevity YYYY-MM-DDTHH:MM

        line = (
            f"• {sym} — {outcome} — {R_txt} — {hours:.1f}h — closed {closed_at}"
        )
        lines.append(line)

    body = "\n".join(lines)

    full_msg = header + "\n" + body

    print("[weekly] Sending Discord weekly recap…")
    post_discord(hook, full_msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Ensure Redis is initialized before load_trades()
    init_redis_from_config(cfg)

    run(cfg)
