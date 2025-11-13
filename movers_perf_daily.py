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
    trades = load_trades()  # movers-only

    if not trades:
        post_discord(hook, "📊 No closed trades yet.")
        return

    today = datetime.datetime.utcnow().date()

    today_trades = [
        t for t in trades
        if t.get("closed_at") and pd.to_datetime(t["closed_at"]).date() == today
    ]

    if not today_trades:
        print("[daily] No trades closed today.")
        return

    print(f"[daily] Found {len(today_trades)} trades closed today.")

    n = len(today_trades)
    wins = [t for t in today_trades if t.get("outcome") in ("t1", "t2")]
    stops = [t for t in today_trades if t.get("outcome") == "stop"]
    win_rate = (len(wins) / n) * 100.0 if n else 0.0

    Rs = [float(t["R"]) for t in today_trades if t.get("R") is not None]
    cum_R = sum(Rs) if Rs else 0.0

    best = max(today_trades, key=lambda t: (t.get("R") or 0.0))
    worst = min(today_trades, key=lambda t: (t.get("R") or 0.0))

    def fmt_R(t):
        r = t.get("R")
        return f"{float(r):+0.2f}R" if r is not None else "n/a"

    msg = (
        f"🗓️ **Daily Movers Recap — {today.isoformat()}**\n"
        f"Closed: **{n}**  |  Wins: **{len(wins)}**  |  Stops: **{len(stops)}**  |  Win rate: **{win_rate:.1f}%**\n"
        f"Total result: **{cum_R:+.2f}R**\n"
        f"Best: `{best['symbol']}` {fmt_R(best)}  |  Worst: `{worst['symbol']}` {fmt_R(worst)}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Recent (today):\n"
    )

    lines = []
    for t in today_trades[-10:]:
        r = t.get("R")
        r_txt = fmt_R(t)
        h = (t.get("time_to_outcome_min", 0.0) or 0.0) / 60.0
        lines.append(
            f"`{t['symbol']}` — {t['outcome'].upper()} — {r_txt} — {h:.1f}h"
        )

    msg += "\n".join(lines)

    print("[daily] Sending Discord daily recap…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_redis_from_config(cfg)
    run(cfg)
