from perf_common import *
import datetime, numpy as np, pandas as pd

def run():
    trades = load_trades()
    if not trades: return
    now = pd.Timestamp.utcnow()
    cutoff = now - pd.Timedelta(days=30)
    recent = [t for t in trades if pd.to_datetime(t["closed_at"]) >= cutoff]
    if not recent: return

    syms = {}
    for t in recent:
        s = t["symbol"]
        syms.setdefault(s, []).append(t)
    leaders = []
    for s, arr in syms.items():
        wins = sum(a["outcome"] in ("t1","t2") for a in arr)
        Rvals = [float(a.get("R",0)) for a in arr if a.get("R") is not None]
        t1s = [a.get("time_to_t1_min") for a in arr if a.get("time_to_t1_min")]
        leaders.append((s, wins, len(arr), sum(Rvals), np.median(t1s)/60 if t1s else 0))
    leaders.sort(key=lambda x: x[3], reverse=True)

    msg = "🏅 **Movers Leaderboard — 30 Days**\n" + "\n".join(
        f"{i+1}) `{s}` — {w}/{n} wins — {r:+.2f}R — med T1: {t:.1f}h"
        for i,(s,w,n,r,t) in enumerate(leaders[:10])
    )
    post_discord(msg)

if __name__ == "__main__":
    run()
