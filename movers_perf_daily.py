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


def summarize(trades):
    """Return (n, win_count, stop_count, win_rate_pct, cum_R)."""
    n = len(trades)
    wins = [t for t in trades if t.get("outcome") in ("t1", "t2")]
    stops = [t for t in trades if t.get("outcome") == "stop"]
    Rs = [float(t["R"]) for t in trades if t.get("R") is not None]
    cum_R = sum(Rs) if Rs else 0.0
    win_rate = (len(wins) / n * 100.0) if n > 0 else 0.0
    return n, len(wins), len(stops), win_rate, cum_R


def fmt_trade_line(t):
    """Format a single trade row for Discord."""
    sym = t.get("symbol", "?")
    outcome = (t.get("outcome") or "").upper()
    R = t.get("R")
    R_txt = f"{float(R):+0.2f}R" if R is not None else "n/a"
    mins = float(t.get("time_to_outcome_min") or 0.0)
    hours = mins / 60.0
    return f"- `{sym}` — {outcome} — {R_txt} — {hours:.1f}h to outcome"


def run(cfg):
    print("[daily] Starting daily movers performance recap...")
    # Init Discord & Redis (Redis is initialized for side-effects + logging)
    hook = init_discord_from_config(cfg)
    _ = init_redis_from_config(cfg)

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
        # You can choose to post or stay silent; here we do a light ping:
        post_discord(hook, f"🗓️ **Daily Movers Recap — {today.isoformat()}**\nNo trades closed today.")
        return

    # ---- Daily stats ----
    n_day, wins_day, stops_day, win_rate_day, cum_R_day = summarize(today_trades)

    # ---- Lifetime stats (all closed movers trades) ----
    n_all, wins_all, stops_all, win_rate_all, cum_R_all = summarize(trades)

    print(f"[daily] Found {n_day} trades closed today.")
    print(f"[daily] Lifetime closed trades: {n_all}")

    # Best / worst of the day by R
    trades_with_R = [t for t in today_trades if t.get("R") is not None]
    if trades_with_R:
        best = max(trades_with_R, key=lambda t: t.get("R", 0))
        worst = min(trades_with_R, key=lambda t: t.get("R", 0))
        best_txt = f"`{best['symbol']}` {best['R']:+.2f}R"
        worst_txt = f"`{worst['symbol']}` {worst['R']:+.2f}R"
    else:
        best_txt = "n/a"
        worst_txt = "n/a"

    # All today's trades as a list
    trade_lines = [fmt_trade_line(t) for t in today_trades]

    msg = f"""🗓️ **Daily Movers Recap — {today.isoformat()}**

**Today**
• Closed: **{n_day}**  
• Wins: **{wins_day}** | Stops: **{stops_day}**  
• Win rate: **{win_rate_day:.1f}%**  
• Total: **{cum_R_day:+.2f}R**  
• Best: {best_txt} | Worst: {worst_txt}

**Since tracking began**
• Closed: **{n_all}**  
• Wins: **{wins_all}** | Stops: **{stops_all}**  
• Win rate: **{win_rate_all:.1f}%**  
• Total: **{cum_R_all:+.2f}R**

**Today's closed trades**
""" + "\n".join(trade_lines)

    print("[daily] Sending Discord daily recap…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_movers_bot_config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
