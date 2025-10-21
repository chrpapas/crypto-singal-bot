#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys, math, statistics
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd
import ccxt
import yaml
import redis

# =============== Helpers ===============

def safe_get_num(x, default=None):
    """Return a float from x; if x is callable, call it; handle None/NaN."""
    try:
        if callable(x):
            x = x()
        if x is None:
            return default
        f = float(x)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default

def fmt_f(x, digs=6, dash='-'):
    """Format number to fixed digs, or dash if not a number."""
    v = safe_get_num(x, None)
    if v is None:
        return dash
    try:
        return f"{v:.{digs}f}"
    except Exception:
        return dash

def fmt_pct(x, digs=2, dash='-'):
    v = safe_get_num(x, None)
    if v is None:
        return dash
    try:
        return f"{v:.{digs}f}%"
    except Exception:
        return dash

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# =============== Redis persistence ===============

class RedisState:
    def __init__(self, url: str, prefix: str):
        if not url:
            raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL env.")
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
        self.prefix = prefix or "spideybot:v1"

    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])

    def load_performance(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state", "performance"))
        if txt:
            try:
                return json.loads(txt)
            except Exception:
                pass
        return {"open_trades": [], "closed_trades": []}

    def load_active_positions(self) -> Dict[str, Any]:
        txt = self.r.get(self.k("state", "active_positions"))
        return json.loads(txt) if txt else {}

# =============== CCXT wrapper ===============

class ExClient:
    def __init__(self, name: str):
        self.name = name
        self.ex = getattr(ccxt, name)({"enableRateLimit": True})
        self._markets = None

    def load_markets(self):
        if self._markets is None:
            try:
                self._markets = self.ex.load_markets()
            except Exception:
                self._markets = {}
        return self._markets

    def has_pair(self, symbol_pair: str) -> bool:
        mkts = self.load_markets() or {}
        if symbol_pair in mkts: return True
        syms = getattr(self.ex, "symbols", None) or []
        return symbol_pair in syms

    def last_close(self, symbol: str, tf: str) -> float:
        # Fetch last 2 candles and use the latest close
        rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=2)
        if not rows:
            raise RuntimeError(f"No OHLCV for {self.name} {symbol} {tf}")
        return float(rows[-1][4])

# =============== Dataclasses for reporting ===============

@dataclass
class ClosedStats:
    n: int
    win_rate: float
    avg_r: float
    med_r: float
    best_r: float
    worst_r: float
    profit_factor: float
    total_r: float

@dataclass
class OpenMTM:
    exchange: str
    symbol: str
    timeframe: str
    entry: float
    stop: float
    last: float
    risk: float
    R_unrealized: float
    pct_unrealized: float

# =============== Core calculators ===============

def compute_closed_stats(closed: List[Dict[str, Any]]) -> ClosedStats:
    if not closed:
        return ClosedStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, float('nan'), 0.0)

    R_vals = []
    for tr in closed:
        r = safe_get_num(tr.get("r_multiple"), 0.0)
        R_vals.append(r)

    n = len(R_vals)
    wins = [r for r in R_vals if r > 0]
    losses = [r for r in R_vals if r < 0]

    win_rate = 100.0 * (len(wins) / n) if n else 0.0
    avg_r = statistics.mean(R_vals) if R_vals else 0.0
    med_r = statistics.median(R_vals) if R_vals else 0.0
    best_r = max(R_vals) if R_vals else 0.0
    worst_r = min(R_vals) if R_vals else 0.0
    gain = sum(wins) if wins else 0.0
    loss = -sum(losses) if losses else 0.0
    profit_factor = float('inf') if loss == 0 and gain > 0 else (gain / loss if loss > 0 else 0.0)
    total_r = sum(R_vals)

    return ClosedStats(
        n=n,
        win_rate=win_rate,
        avg_r=avg_r,
        med_r=med_r,
        best_r=best_r,
        worst_r=worst_r,
        profit_factor=profit_factor,
        total_r=total_r
    )

def group_by_timeframe(closed: List[Dict[str, Any]]) -> Dict[str, ClosedStats]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for tr in closed:
        tf = tr.get("timeframe") or "?"
        buckets.setdefault(tf, []).append(tr)
    out = {}
    for tf, arr in buckets.items():
        out[tf] = compute_closed_stats(arr)
    return out

def rolling_equity_R(R_vals: List[float]) -> Tuple[List[float], float]:
    """Return rolling equity curve and max drawdown (R)."""
    eq = []
    s = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in R_vals:
        s += r
        eq.append(s)
        peak = max(peak, s)
        dd = s - peak
        if dd < max_dd:
            max_dd = dd
    return eq, max_dd

# =============== Snapshot computation ===============

def compute_snapshot(cfg: Dict[str, Any]):
    # Redis config
    p = cfg.get("persistence", {})
    redis_url = p.get("redis_url") or os.environ.get("REDIS_URL")
    key_prefix = p.get("key_prefix", "spideybot:v1")

    # Optional risk → money conversion
    risk_per_trade_usd = None
    if cfg.get("__cli_risk_per_trade_usd__") is not None:
        risk_per_trade_usd = float(cfg["__cli_risk_per_trade_usd__"])

    rs = RedisState(redis_url, key_prefix)
    perf = rs.load_performance()

    closed = perf.get("closed_trades", []) or []
    open_trs = perf.get("open_trades", []) or []

    print("=== Performance Snapshot ===")
    print(f"As of: {pd.Timestamp.utcnow().isoformat()}Z")
    print(f"Redis prefix: {key_prefix}")
    print()

    # ----- Closed trades -----
    print("--- Closed trades ---")
    cs = compute_closed_stats(closed)
    print(f"Closed trades: {cs.n}")
    print(f"Win rate:      {cs.win_rate:.1f}%")
    print(f"Avg R:         {cs.avg_r:.3f}   | Median R: {cs.med_r:.3f}")
    print(f"Best/Worst R:  {cs.best_r:.2f} / {cs.worst_r:.2f}")
    pf_txt = "inf" if math.isinf(cs.profit_factor) else f"{cs.profit_factor:.2f}"
    print(f"Profit factor: {pf_txt}")
    print()

    # ----- Open trades MTM -----
    print("--- Open trades (MTM) ---")
    # Build exchange clients that are actually needed
    ex_names_needed = sorted(set([t.get("exchange") for t in open_trs if t.get("exchange")]))
    ex_clients: Dict[str, ExClient] = {}
    for ex_name in ex_names_needed:
        try:
            ex_clients[ex_name] = ExClient(ex_name)
        except Exception as e:
            print(f"[ex] {ex_name} init err:", e)

    open_mtm: List[OpenMTM] = []
    for tr in open_trs:
        if tr.get("status") != "open":
            continue
        ex_name = tr.get("exchange")
        sym = tr.get("symbol")
        tf = tr.get("timeframe") or "1h"
        entry = safe_get_num(tr.get("entry"), None)
        stop = safe_get_num(tr.get("stop"), None)
        risk = safe_get_num(tr.get("risk"), None)

        if ex_name not in ex_clients or entry is None or stop is None or risk is None or risk <= 0:
            # Skip malformed trade
            continue
        try:
            last = ex_clients[ex_name].last_close(sym, tf)
        except Exception as e:
            # If fetch fails, skip this trade (or use entry as proxy)
            last = entry

        R_unreal = (last - entry) / max(risk, 1e-12)
        pct_unreal = (last / entry - 1.0) * 100.0 if entry else 0.0
        open_mtm.append(OpenMTM(
            exchange=ex_name, symbol=sym, timeframe=tf,
            entry=float(entry), stop=float(stop), last=float(last),
            risk=float(risk), R_unrealized=float(R_unreal),
            pct_unrealized=float(pct_unreal)
        ))

    unreal_R_total = sum([t.R_unrealized for t in open_mtm]) if open_mtm else 0.0
    print(f"Open trades:   {len(open_mtm)}")
    print(f"Unrealized R:  {unreal_R_total:.3f}")
    if risk_per_trade_usd is not None:
        print(f"Unrealized P/L: ${unreal_R_total * risk_per_trade_usd:.2f} (assumes ${risk_per_trade_usd:.2f} risk/trade)")
    print()

    # ----- Equity curves in R -----
    print("--- Equity (R) ---")
    closed_R = [safe_get_num(t.get("r_multiple"), 0.0) for t in closed]
    eq_closed, max_dd_closed = rolling_equity_R(closed_R)
    total_closed = eq_closed[-1] if eq_closed else 0.0
    print(f"Equity (closed only): {total_closed:.2f} R | Max DD: {max_dd_closed:.2f} R")

    eq_plus_open = total_closed + unreal_R_total
    # A conservative open-DD proxy: min(0, unreal_R_total)
    max_dd_plus_open = min(max_dd_closed, max_dd_closed + unreal_R_total)
    print(f"Equity (+open MTM):   {eq_plus_open:.2f} R | Max DD: {max_dd_plus_open:.2f} R")

    if risk_per_trade_usd is not None:
        print(f"Equity (closed): ${total_closed * risk_per_trade_usd:.2f}")
        print(f"Equity (+open):  ${eq_plus_open * risk_per_trade_usd:.2f}")
    print()

    # ----- Breakdown by timeframe -----
    print("--- By timeframe ---")
    by_tf = group_by_timeframe(closed)
    for tf, st in sorted(by_tf.items()):
        pf_txt = "inf" if math.isinf(st.profit_factor) else f"{st.profit_factor:.2f}"
        print(f" {tf}: n={st.n} | win {st.win_rate:.1f}% | avgR {st.avg_r:.3f} | PF {pf_txt}")
    print()

    # ----- Open trades detail -----
    if open_mtm:
        print("--- Open trades detail (top 10 by |R|) ---")
        top = sorted(open_mtm, key=lambda t: abs(t.R_unrealized), reverse=True)[:10]
        for r in top:
            print(
                f"[{r.exchange}] {r.symbol} {r.timeframe} | "
                f"R={fmt_f(r.R_unrealized,2)} | {fmt_f(r.pct_unrealized,2)}% | "
                f"entry={fmt_f(r.entry,6)} last={fmt_f(r.last,6)}"
            )
        print()

    # ----- Totals in money (optional) -----
    if risk_per_trade_usd is not None:
        total_money_closed = total_closed * risk_per_trade_usd
        total_money_open = unreal_R_total * risk_per_trade_usd
        print("--- Money P/L (assumed) ---")
        print(f"Closed P/L:  ${total_money_closed:.2f}")
        print(f"Open P/L:    ${total_money_open:.2f}")
        print(f"Net P/L:     ${total_money_closed + total_money_open:.2f}")

# =============== Entrypoint ===============

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--risk_per_trade_usd", type=float, default=None,
                    help="If set, also show money P/L by multiplying R with this per-trade risk (e.g., 100).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # expand ${VAR}
    def expand_env(obj):
        if isinstance(obj, dict): return {k: expand_env(v) for k, v in obj.items()}
        if isinstance(obj, list): return [expand_env(x) for x in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], obj)
        return obj

    cfg = expand_env(cfg)
    if args.risk_per_trade_usd is not None:
        cfg["__cli_risk_per_trade_usd__"] = args.risk_per_trade_usd

    try:
        compute_snapshot(cfg)
    except Exception as e:
        print("Snapshot failed:", e)
        sys.exit(1)