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
    cutoff = today - datetime.timedelta(days=7)

    week_trades = []
    for t in trades:
        ca = t.get("closed_at")
        if not ca:
            continue
        d = pd.to_datetime(ca).date()
        if d >= cutoff:
            week_trades.append(t)

    if not week_trades:
        print("[weekly] No trades in last 7 days.")
        return

    print(f"[weekly] Found {len(week_trades)} trades closed in last 7 days.")

    n = len(week_trades)
    wins = [t for t in week_trades if t.get("outcome") in ("t1", "t2")]
    stops = [t for t in week_trades if t.get("outcome") == "stop"]
    win_rate = (len(wins) / n) * 100.0 if n else 0.0

    Rs = [float(t["R"]) for t in week_trades if t.get("R") is not None]
    cum_R = sum(Rs) if Rs else 0.0

    best = max(week_trades, key=lambda t: (t.get("R") or 0.0))
    worst = min(week_trades, key=lambda t: (t.get("R") or 0.0))

    def fmt_R(t):
        r = t.get("R")
        return f"{float(r):+0.2f}R" if r is not None else "n/a"

    msg = (
        f"📆 **Weekly Movers Recap — last 7 days (since {cutoff.isoformat()})**\n"
        f"Closed: **{n}**  |  Wins: **{len(wins)}**  |  Stops: **{len(stops)}**  |  Win rate: **{win_rate:.1f}%**\n"
        f"Total result: **{cum_R:+.2f}R**\n"
        f"Best: `{best['symbol']}` {fmt_R(best)}  |  Worst: `{worst['symbol']}` {fmt_R(worst)}\n"
        f"━━━━━━━━━━━━━━\n"
        f"Recent (last 10 closed):\n"
    )

    # Sort by closed_at ascending, then take last 10
    week_trades_sorted = sorted(
        week_trades,
        key=lambda t: pd.to_datetime(t.get("closed_at") or "1970-01-01")
    )

    lines = []
    for t in week_trades_sorted[-10:]:
        r_txt = fmt_R(t)
        d = pd.to_datetime(t["closed_at"]).date()
        h = (t.get("time_to_outcome_min", 0.0) or 0.0) / 60.0
        lines.append(
            f"`{t['symbol']}` — {t['outcome'].upper()} — {r_txt} — {h:.1f}h — {d.isoformat()}"
        )

    msg += "\n".join(lines)

    print("[weekly] Sending Discord weekly recap…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_redis_from_config(cfg)
    run(cfg)
