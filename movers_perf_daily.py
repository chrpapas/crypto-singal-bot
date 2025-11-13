from perf_common import *
import datetime

def run():
    trades = load_trades()
    if not trades:
        post_discord("📊 No closed trades yet.")
        return

    today = datetime.datetime.utcnow().date()
    today_trades = [
        t for t in trades
        if t.get("closed_at") and pd.to_datetime(t["closed_at"]).date() == today
    ]
    if not today_trades:
        print("No trades today.")
        return

    n = len(today_trades)
    wins = [t for t in today_trades if t["outcome"] in ("t1","t2")]
    stops = [t for t in today_trades if t["outcome"] == "stop"]
    win_rate = (len(wins)/n)*100

    Rs = [float(t.get("R",0)) for t in today_trades if t.get("R") is not None]
    cum_R = sum(Rs)
    best = max(today_trades, key=lambda t: t.get("R",0))
    worst = min(today_trades, key=lambda t: t.get("R",0))

    msg = f"""🗓️ **Daily Movers Recap — {today.isoformat()}**
Closed: **{n}**  |  Win rate: **{win_rate:.1f}%**  |  Total: **{cum_R:+.2f}R**
Best: `{best['symbol']}` {best['R']:+.2f}R  |  Worst: `{worst['symbol']}` {worst['R']:+.2f}R
━━━━━━━━━━━━━━
Recent:
""" + "\n".join(
        f"{t['symbol']} — {t['outcome'].upper()} — {t['R']:+.2f}R — {(t.get('time_to_outcome_min',0)/60):.1f}h"
        for t in today_trades[-10:]
    )
    post_discord(msg)

if __name__ == "__main__":
    run()
