#!/usr/bin/env python3
import argparse, datetime
from collections import defaultdict

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
        print("[leaderboard] No trades in last 30 days.")
        return

    print(f"[leaderboard] Building leaderboard from {len(recent)} trades in last 30 days.")

    # Aggregate by symbol
    stats = defaultdict(lambda: {
        "n": 0,
        "wins": 0,
        "stops": 0,
        "total_R": 0.0,
    })

    for t in recent:
        sym = t.get("symbol", "UNKNOWN")
        s = stats[sym]
        s["n"] += 1
        if t.get("outcome") in ("t1", "t2"):
            s["wins"] += 1
        if t.get("outcome") == "stop":
            s["stops"] += 1
        if t.get("R") is not None:
            s["total_R"] += float(t["R"])

    # Create list with computed winrate
    rows = []
    for sym, s in stats.items():
        n = s["n"]
        win_rate = (s["wins"] / n) * 100.0 if n else 0.0
        rows.append({
            "symbol": sym,
            "n": n,
            "wins": s["wins"],
            "stops": s["stops"],
            "total_R": s["total_R"],
            "win_rate": win_rate,
        })

    # Top by total_R
    top_total = sorted(rows, key=lambda r: r["total_R"], reverse=True)[:10]
    # Most consistent (min 3 trades) by win_rate
    candidates = [r for r in rows if r["n"] >= 3]
    top_consistent = sorted(candidates, key=lambda r: r["win_rate"], reverse=True)[:10]

    msg = (
        f"🏆 **Movers Leaderboard — last 30 days (since {cutoff.isoformat()})**\n"
        f"Symbols with most closed trades: **{len(rows)}**\n"
        f"━━━━━━━━━━━━━━\n"
        f"**Top by total R (min 1 trade)**\n"
    )

    if not top_total:
        msg += "_No symbols yet._\n"
    else:
        for r in top_total:
            msg += (
                f"`{r['symbol']}` — {r['total_R']:+.2f}R  "
                f"({r['n']} trades, {r['win_rate']:.0f}% wins)\n"
            )

    msg += "\n**Most consistent (min 3 trades)**\n"
    if not top_consistent:
        msg += "_No symbols meet the 3-trade minimum yet._"
    else:
        for r in top_consistent:
            msg += (
                f"`{r['symbol']}` — {r['win_rate']:.0f}% wins  "
                f"({r['n']} trades, {r['total_R']:+.2f}R)\n"
            )

    print("[leaderboard] Sending Discord leaderboard…")
    post_discord(hook, msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="movers-signals-config.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_redis_from_config(cfg)
    run(cfg)
