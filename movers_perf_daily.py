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
    # Init Redis + Discord
    r, prefix = init_redis_from_config(cfg)
    hook = init_discord_from_config(cfg)

    trades = load_trades(r, prefix)

    if not trades:
        print("[daily] No closed trades found.")
        post_discord(hook, "📊 No closed Movers trades yet.")
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
    win_rate = (len(wins) / n) * 100.0

    Rs = [float(t.get("R", 0.0)) for t in today_trades if t.get("R") is not None]
    cum_R = sum(Rs) if Rs else 0.0

    best = max(today_trades, key=lambda t: float(t.get("R", 0.0)))
    worst = min(today_trades, key=lambda t: float(t.get("R", 0.0)))

    def fmt_R(t):
        r = float(t.get("R", 0.0))
        return f"{r:+.2f}R"

    msg = (
        f"🗓️ **Daily Movers Recap — {today.isoformat()}**\n"
        f"Closed: **{n}** | Wins: **{len(wins)}** | Stops: **{len(stops)}** | Win rate: **{win_rate:.1f}%**\n"
        f"Total: **{cum_R:+.2f}R**\n"
        f"Best: `{best['symbol']}` {fmt_R(best)} | Worst: `{worst['symbol']}` {fmt_R(worst)}\n"
        "━━━━━━━━━━━━━━\n"
        "Recent:\n"
    )

    recent_lines = []
    for t in today_trades[-10:]:
        sym = t.get("symbol", "?")
        outcome = (t.get("outcome") or "").upper()
        R_val = float(t.get("R", 0.0)) if t.get("R") is not None else 0.0
        mins = float(t.get("time_to_outcome_min", 0.0) or 0.0)
        hours = mins / 60.0
        recent_lines.append(f"{sym} — {outcome} — {R_val:+.2f}R — {hours:.1f}h")

    msg += "\n".join(recent_lines)

    print("[daily] Sending Discord daily recap…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_movers_bot_config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
