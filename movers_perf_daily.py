from perf_common import *
import datetime
import pandas as pd  # <-- you were missing this import


def run():
    print("[daily] Starting Daily Movers Recap...")

    trades = load_trades()
    if not trades:
        print("[daily] No closed trades in Redis yet.")
        post_discord("📊 Daily Movers Recap\nNo closed trades recorded yet.")
        return

    today = datetime.datetime.utcnow().date()
    today_trades = []

    # Safely collect trades closed today
    for t in trades:
        closed_at = t.get("closed_at")
        if not closed_at:
            continue
        try:
            d = pd.to_datetime(closed_at).date()
        except Exception:
            continue
        if d == today:
            today_trades.append(t)

    if not today_trades:
        print(f"[daily] No trades closed today ({today.isoformat()}).")
        # Optional: also send a tiny Discord notice instead of silence
        post_discord(f"🗓️ **Daily Movers Recap — {today.isoformat()}**\nNo movers closed today.")
        return

    n = len(today_trades)
    print(f"[daily] Found {n} trades closed today.")

    wins = [t for t in today_trades if t.get("outcome") in ("t1", "t2")]
    stops = [t for t in today_trades if t.get("outcome") == "stop"]
    win_rate = (len(wins) / n) * 100 if n > 0 else 0.0

    # Collect valid R values
    trades_with_R = [t for t in today_trades if t.get("R") is not None]
    Rs = [float(t["R"]) for t in trades_with_R]
    cum_R = sum(Rs) if Rs else 0.0

    # Best / worst by R (if we have any R)
    best = max(trades_with_R, key=lambda t: float(t["R"])) if trades_with_R else None
    worst = min(trades_with_R, key=lambda t: float(t["R"])) if trades_with_R else None

    def fmt_R(val):
        try:
            return f"{float(val):+.2f}R"
        except Exception:
            return "n/a"

    # Build message
    header = f"🗓️ **Daily Movers Recap — {today.isoformat()}**"
    summary = (
        f"Closed: **{n}**  |  "
        f"Win rate: **{win_rate:.1f}%**  |  "
        f"Total: **{cum_R:+.2f}R**"
    )

    if best and worst:
        best_str = f"`{best['symbol']}` {fmt_R(best.get('R'))}"
        worst_str = f"`{worst['symbol']}` {fmt_R(worst.get('R'))}"
    else:
        best_str = "n/a"
        worst_str = "n/a"

    bw_line = f"Best: {best_str}  |  Worst: {worst_str}"

    recent_lines = []
    for t in today_trades[-10:]:
        sym = t.get("symbol", "?")
        outcome = (t.get("outcome") or "").upper() or "?"
        R_val = fmt_R(t.get("R"))
        mins = t.get("time_to_outcome_min", 0) or 0
        try:
            hours = float(mins) / 60.0
        except Exception:
            hours = 0.0
        recent_lines.append(
            f"{sym} — {outcome} — {R_val} — {hours:.1f}h"
        )

    msg = (
        f"{header}\n"
        f"{summary}\n"
        f"{bw_line}\n"
        "━━━━━━━━━━━━━━\n"
        "Recent:\n" +
        "\n".join(recent_lines)
    )

    print("[daily] Posting daily recap to Discord.")
    post_discord(msg)


if __name__ == "__main__":
    run()
