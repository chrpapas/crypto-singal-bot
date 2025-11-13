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
    closed_at = t.get("closed_at", "")[:19]
    return f"- `{sym}` — {outcome} — {R_txt} — {hours:.1f}h — closed {closed_at} UTC"


def run(cfg):
    print("[weekly] Starting weekly movers performance recap...")
    hook = init_discord_from_config(cfg)
    _ = init_redis_from_config(cfg)

    trades = load_trades()
    if not trades:
        print("[weekly] No closed trades found in Redis.")
        post_discord(hook, "📊 No closed trades yet.")
        return

    today = datetime.datetime.utcnow().date()
    week_start = today - datetime.timedelta(days=6)  # last 7 days inclusive
    week_end = today

    weekly_trades = []
    for t in trades:
        ca = t.get("closed_at")
        if not ca:
            continue
        d = pd.to_datetime(ca).date()
        if week_start <= d <= week_end:
            weekly_trades.append(t)

    if not weekly_trades:
        print(f"[weekly] No trades closed between {week_start} and {week_end}.")
        post_discord(
            hook,
            f"📊 **Weekly Movers Recap — {week_start.isoformat()} → {week_end.isoformat()}**\nNo trades closed this week."
        )
        return

    # ---- Weekly stats ----
    n_week, wins_week, stops_week, win_rate_week, cum_R_week = summarize(weekly_trades)

    # ---- Lifetime stats ----
    n_all, wins_all, stops_all, win_rate_all, cum_R_all = summarize(trades)

    print(f"[weekly] Found {n_week} trades closed this week.")
    print(f"[weekly] Lifetime closed trades: {n_all}")

    trades_with_R = [t for t in weekly_trades if t.get("R") is not None]
    if trades_with_R:
        best = max(trades_with_R, key=lambda t: t.get("R", 0))
        worst = min(trades_with_R, key=lambda t: t.get("R", 0))
        best_txt = f"`{best['symbol']}` {best['R']:+.2f}R"
        worst_txt = f"`{worst['symbol']}` {worst['R']:+.2f}R"
    else:
        best_txt = "n/a"
        worst_txt = "n/a"

    trade_lines = [fmt_trade_line(t) for t in weekly_trades]

    msg = f"""📊 **Weekly Movers Recap — {week_start.isoformat()} → {week_end.isoformat()}**

**This week**
• Closed: **{n_week}**  
• Wins: **{wins_week}** | Stops: **{stops_week}**  
• Win rate: **{win_rate_week:.1f}%**  
• Total: **{cum_R_week:+.2f}R**  
• Best: {best_txt} | Worst: {worst_txt}

**Since tracking began**
• Closed: **{n_all}**  
• Wins: **{wins_all}** | Stops: **{stops_all}**  
• Win rate: **{win_rate_all:.1f}%**  
• Total: **{cum_R_all:+.2f}R**

**All trades closed this week**
""" + "\n".join(trade_lines)

    print("[weekly] Sending Discord weekly recap…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_movers_bot_config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
