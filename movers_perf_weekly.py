from perf_common import *
import datetime, numpy as np

def run():
    trades = load_trades()
    if not trades: return
    now = datetime.datetime.utcnow()
    week_ago = now - datetime.timedelta(days=7)
    week_trades = [
        t for t in trades if pd.to_datetime(t["closed_at"]) >= week_ago
    ]
    if not week_trades: return

    n = len(week_trades)
    n_win = sum(t["outcome"] in ("t1","t2") for t in week_trades)
    n_stop = sum(t["outcome"] == "stop" for t in week_trades)
    wr = (n_win/n)*100

    times = [t.get("time_to_outcome_min") for t in week_trades if t.get("time_to_outcome_min")]
    tmed = np.median(times)/60 if times else 0
    tmean = np.mean(times)/60 if times else 0

    Rs = [float(t.get("R",0)) for t in week_trades if t.get("R") is not None]
    cumR = sum(Rs)
    msg = f"""📈 **Kriticurrency Weekly Scorecard**
Period: {week_ago.date()} → {now.date()}
Closed: **{n}**  |  Wins: **{n_win}**  |  Stops: **{n_stop}**
Win rate: **{wr:.1f}%**  |  Total: **{cumR:+.2f}R**
Median time to outcome: **{tmed:.1f}h** (avg {tmean:.1f}h)
━━━━━━━━━━━━━━
Top 5 symbols:
""" + "\n".join(
        f"• {sym}: {cnt} trades"
        for sym, cnt in sorted(
            ((s, sum(t['symbol']==s for t in week_trades)) for s in set(t['symbol'] for t in week_trades)),
            key=lambda x: x[1], reverse=True
        )[:5]
    )
    post_discord(msg)

if __name__ == "__main__":
    run()
