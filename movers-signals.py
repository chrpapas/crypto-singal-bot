#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mexc_movers_bot.py — Movers-only scanner with:
- CMC Movers universe -> mapped to MEXC spot
- Legacy Movers signal logic (5% / 10% targets)
- Redis-based silent registry + performance stats
- Discord: Movers signals only (no MEXC mention in message)

ENV (optional):
  REDIS_URL, DISCORD_SIGNALS_WEBHOOK, CMC_API_KEY

Run:
  python3 mexc_movers_bot.py --config mexc_movers_bot_config.yml
"""

import argparse, json, os, yaml, requests
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import ccxt
import redis

# =============== TA helpers ===============
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(length).mean()
    dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast)
    s = ema(series, slow)
    line = f - s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# =============== Exchange wrapper (MEXC) ===============
class ExClient:
    def __init__(self):
        self.name = "mexc"
        self.ex = ccxt.mexc({"enableRateLimit": True})
        key = os.environ.get("MEXC_API_KEY")
        sec = os.environ.get("MEXC_SECRET")
        if key and sec:
            self.ex.apiKey = key
            self.ex.secret = sec
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
        if symbol_pair in mkts:
            return True
        syms = getattr(self.ex, "symbols", None) or []
        return symbol_pair in syms

    def ohlcv(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    def last_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.ex.fetch_ticker(symbol)
            px = t.get("last") or t.get("close")
            return float(px) if px is not None else None
        except Exception:
            try:
                df = self.ohlcv(symbol, "1h", 2)
                return float(df["close"].iloc[-1])
            except Exception:
                return None

# =============== Redis persistence ===============
class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        if not url:
            raise RuntimeError("Redis URL missing. Provide persistence.redis_url or REDIS_URL.")
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48 * 3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")

    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])

    def get(self, *parts) -> Optional[str]:
        return self.r.get(self.k(*parts))

    def set(self, val: str, *parts):
        self.r.set(self.k(*parts), val)

    def setex(self, val: str, *parts):
        self.r.setex(self.k(*parts), self.ttl_seconds, val)

    def load_json(self, *parts, default=None):
        txt = self.get(*parts)
        if not txt:
            return default if default is not None else {}
        try:
            return json.loads(txt)
        except Exception:
            return default if default is not None else {}

    def save_json(self, obj, *parts):
        self.set(json.dumps(obj), *parts)

# =============== Stablecoin filtering (base-only) ===============
DEFAULT_STABLES = {
    "USDT","USDC","USDD","DAI","FDUSD","TUSD","BUSD","PAX","PAXG","XAUT","USD1","USDE","USDY",
    "USDP","SUSD","EURS","EURT","PYUSD"
}

def is_stable_or_pegged(symbol: str, extra: List[str]) -> bool:
    base, _ = symbol.split("/")
    b = base.upper().replace("3L", "").replace("3S", "").replace("5L", "").replace("5S", "")
    extras = {e.upper() for e in (extra or [])}
    return (b in DEFAULT_STABLES) or (b in extras)

# =============== CMC Universe (Movers) ===============
def fetch_cmc_listings(cfg: Dict[str, Any], limit=500) -> List[dict]:
    headers = {
        "X-CMC_PRO_API_KEY": os.environ.get("CMC_API_KEY") or cfg.get("movers", {}).get("cmc_api_key", "")
    }
    if not headers["X-CMC_PRO_API_KEY"]:
        return []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {"limit": limit, "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        return r.json().get("data", [])
    except Exception:
        return []

def cmc_movers_symbols(cfg: Dict[str, Any]) -> List[str]:
    mv = cfg.get("movers", {})
    data = fetch_cmc_listings(cfg, limit=mv.get("limit", 500))
    out = []
    now = pd.Timestamp.utcnow()
    min_change = mv.get("min_change_24h", 15.0)
    min_vol = mv.get("min_volume_usd_24h", 5_000_000)
    max_age = mv.get("max_age_days", 365)
    for it in data:
        sym = it.get("symbol", "").upper()
        q = it.get("quote", {}).get("USD", {})
        ch = (q.get("percent_change_24h") or 0.0)
        vol = (q.get("volume_24h") or 0.0)
        date_added = pd.to_datetime(it.get("date_added", now.isoformat()), utc=True)
        age_days = (now - date_added).days
        if (ch >= min_change) and (vol >= min_vol) and (age_days <= max_age):
            out.append(sym)
    return out

def filter_pairs_on_mexc(client: ExClient, symbols: List[str], quote: str = "USDT", extra_stables: List[str] = None) -> List[str]:
    client.load_markets()
    out = []
    for sym in symbols:
        pair = f"{sym}/{quote}"
        if client.has_pair(pair) and not is_stable_or_pegged(pair, extra_stables or []):
            out.append(pair)
    return out

# =============== Legacy Movers Signal (5% / 10% targets) ===============
def legacy_mover_signal(df1h: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df1h is None or len(df1h) < 80:
        return None

    e20, e50 = ema(df1h["close"], 20), ema(df1h["close"], 50)
    mac_line, mac_sig, _ = macd(df1h["close"])
    r = rsi(df1h["close"], 14)
    vS = sma(df1h["volume"], 20)

    last = df1h.iloc[-1]
    hl = df1h["high"].iloc[-31:-1].max()

    aligned = (
        e20.iloc[-1] > e50.iloc[-1] and
        mac_line.iloc[-1] > mac_sig.iloc[-1] and
        55 <= r.iloc[-1] <= 80
    )

    breakout = (
        last["close"] > hl and
        last["volume"] >= (vS.iloc[-1] if not np.isnan(vS.iloc[-1]) else last["volume"])
    )

    if not (aligned and breakout):
        return None

    entry = float(last["close"])
    stop = float(min(df1h["low"].iloc[-10:]))

    return {
        "type": "day",
        "entry": entry,
        "stop": stop,
        "t1": round(entry * 1.05, 6),  # +5%
        "t2": round(entry * 1.10, 6),  # +10%
        "level": float(hl),
        "note": "Mover Trend",
        "event_bar_ts": df1h.index[-1].isoformat(),
    }

# =============== Silent registry (Movers) ===============
def silent_key(symbol: str, tf: str, typ: str, source: str) -> str:
    return f"{symbol}|{tf}|{typ}|{source}"

def record_new_silent(rds: RedisState, sig: Dict[str, Any], source: str):
    book = rds.load_json("state", "silent_open", default={})
    k = silent_key(sig["symbol"], sig["timeframe"], sig["type"], source)
    book[k] = {
        "symbol": sig["symbol"],
        "tf": sig["timeframe"],
        "type": sig["type"],
        "source": source,
        "entry": float(sig["entry"]),
        "stop": float(sig["stop"]) if sig.get("stop") is not None else None,
        "t1": float(sig.get("t1")) if sig.get("t1") is not None else None,
        "t2": float(sig.get("t2")) if sig.get("t2") is not None else None,
        "opened_at": sig.get("event_bar_ts"),
        "status": "open",
        "bar_id": sig.get("event_bar_ts"),
    }
    rds.save_json(book, "state", "silent_open")

def is_silent_open_movers(rds: RedisState, symbol: str, tf: str, typ: str) -> bool:
    book = rds.load_json("state", "silent_open", default={})
    return silent_key(symbol, tf, typ, "movers") in book

def close_silent_if_hit(rds: RedisState, client: ExClient, cfg: Dict[str, Any]):
    """
    Evaluate all open silent signals and close them if stop/T1/T2 is hit.
    Also logs:
      - time_to_outcome_min
      - time_to_t1_min
      - time_to_t2_min
    in minutes, based on OHLCV timestamps.
    """
    book = rds.load_json("state", "silent_open", default={})
    if not book:
        return
    changed = False

    for k, tr in list(book.items()):
        if tr.get("status") != "open":
            continue

        pair = tr["symbol"]
        tf = tr["tf"]
        stop = tr.get("stop")
        t1 = tr.get("t1")
        t2 = tr.get("t2")

        # Movers only: no bear logic here
        try:
            limit = 400 if tf in ("1h", "4h") else 330
            df = client.ohlcv(pair, tf, limit)
            opened_at = pd.to_datetime(tr["opened_at"], utc=True)
            df = df.loc[opened_at:]
            rows = df.iloc[1:]  # bars after signal bar

            outcome = None
            price = None
            hit_ts = None

            # first time T1 is ever touched (even if we later hit T2)
            t1_ts = None

            for ts, r in rows.iterrows():
                lo, hi = float(r["low"]), float(r["high"])

                # track first touch of T1
                if t1 is not None and hi >= t1 and t1_ts is None:
                    t1_ts = ts

                # both stop and targets in same candle -> optimistic ordering
                if stop is not None and lo <= stop and t2 is not None and hi >= t2:
                    outcome, price, hit_ts = "t2", t2, ts
                    break
                if stop is not None and lo <= stop and t1 is not None and hi >= t1:
                    outcome, price, hit_ts = "t1", t1, ts
                    break

                # simple cases
                if stop is not None and lo <= stop:
                    outcome, price, hit_ts = "stop", stop, ts
                    break
                if t2 is not None and hi >= t2:
                    outcome, price, hit_ts = "t2", t2, ts
                    break
                if t1 is not None and hi >= t1:
                    outcome, price, hit_ts = "t1", t1, ts
                    break

            if outcome:
                tr["status"] = "closed"
                tr["closed_at"] = hit_ts.isoformat() if hit_ts is not None else df.index[-1].isoformat()
                tr["outcome"] = outcome  # "t1", "t2" or "stop"
                tr["exit_price"] = float(price)

                # timing metrics
                if hit_ts is not None:
                    dt_outcome = (hit_ts - opened_at).total_seconds() / 60.0
                    tr["time_to_outcome_min"] = float(dt_outcome)
                else:
                    tr["time_to_outcome_min"] = None

                if t1_ts is not None:
                    dt_t1 = (t1_ts - opened_at).total_seconds() / 60.0
                    tr["time_to_t1_min"] = float(dt_t1)
                else:
                    tr["time_to_t1_min"] = None

                if outcome == "t2" and hit_ts is not None:
                    dt_t2 = (hit_ts - opened_at).total_seconds() / 60.0
                    tr["time_to_t2_min"] = float(dt_t2)
                else:
                    tr["time_to_t2_min"] = None

                # optional: compute R multiple
                try:
                    entry = float(tr.get("entry"))
                    stop_val = tr.get("stop")
                    if stop_val is not None:
                        stop_val = float(stop_val)
                        risk = max(1e-9, entry - stop_val)
                        tr["R"] = (tr["exit_price"] - entry) / risk
                    else:
                        tr["R"] = None
                except Exception:
                    tr["R"] = None

                book[k] = tr
                changed = True

        except Exception as e:
            print("[silent] eval err", pair, tf, e)

    if changed:
        rds.save_json(book, "state", "silent_open")

def movers_stats_table(rds: RedisState):
    book = rds.load_json("state", "silent_open", default={})
    movers = [
        tr for tr in book.values()
        if tr.get("source") == "movers" and tr.get("status") == "closed"
    ]
    n = len(movers)
    if n == 0:
        print("--- Movers Performance: no closed trades yet ---")
        return

    n_t1 = sum(1 for tr in movers if tr.get("outcome") == "t1")
    n_t2 = sum(1 for tr in movers if tr.get("outcome") == "t2")
    n_stop = sum(1 for tr in movers if tr.get("outcome") == "stop")

    win_total = n_t1 + n_t2

    def pct(x): return (x / n) * 100.0 if n else 0.0

    print("--- Movers Performance (Signals Source = 'movers') ---")
    print(f"Total closed trades : {n}")
    print(f"Wins (T1 or T2)     : {win_total}  ({pct(win_total):5.1f}%)")
    print(f"  • T1 only         : {n_t1}      ({pct(n_t1):5.1f}%)")
    print(f"  • T2 hits         : {n_t2}      ({pct(n_t2):5.1f}%)")
    print(f"Losses (Stop)       : {n_stop}    ({pct(n_stop):5.1f}%)")
    print("-------------------------------------------------")

def movers_time_stats_table(rds: RedisState):
    """
    Time-to-target stats for Movers signals.

    Uses:
      - time_to_outcome_min when available
      - otherwise falls back to (closed_at - opened_at) if both are present
    """
    book = rds.load_json("state", "silent_open", default={})

    movers = [
        tr for tr in book.values()
        if tr.get("source") == "movers" and tr.get("status") == "closed"
    ]
    if not movers:
        print("--- Movers Time Stats: no closed movers trades yet ---")
        return

    def _effective_outcome_minutes(tr):
        v = tr.get("time_to_outcome_min")
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
        # fallback: derive from opened_at / closed_at
        oa = tr.get("opened_at")
        ca = tr.get("closed_at")
        if not oa or not ca:
            return None
        try:
            oa_dt = pd.to_datetime(oa, utc=True)
            ca_dt = pd.to_datetime(ca, utc=True)
            return float((ca_dt - oa_dt).total_seconds() / 60.0)
        except Exception:
            return None

    # any closed trade where outcome is t1 or t2 or stop
    outcome_any = [
        _effective_outcome_minutes(tr)
        for tr in movers
    ]
    outcome_any = [v for v in outcome_any if v is not None]

    # For T1/T2 we still rely only on explicit timing fields
    def _clean(values):
        return [float(v) for v in values if v is not None]

    t1_any = _clean(
        tr.get("time_to_t1_min")
        for tr in movers
        if tr.get("outcome") in ("t1", "t2")
    )

    t2_only = _clean(
        tr.get("time_to_t2_min")
        for tr in movers
        if tr.get("outcome") == "t2"
    )

    if not (t1_any or t2_only or outcome_any):
        print("--- Movers Time Stats: no timing data yet ---")
        return

    def _stats(arr):
        if not arr:
            return None
        a = np.array(arr, dtype=float)
        return {
            "n": len(a),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
        }

    def _fmt_block(title, st):
        if not st:
            print(f"{title}: no samples")
            return

        def _h(m): return m / 60.0

        print(f"{title}: n={st['n']}")
        print(f"  • min    : {st['min']:.1f} min  ({_h(st['min']):.2f} h)")
        print(f"  • median : {st['median']:.1f} min  ({_h(st['median']):.2f} h)")
        print(f"  • mean   : {st['mean']:.1f} min  ({_h(st['mean']):.2f} h)")
        print(f"  • max    : {st['max']:.1f} min  ({_h(st['max']):.2f} h)")

    print("--- Movers Time-To-Target Stats ---")
    _fmt_block("Time to first outcome (T1/T2/Stop)", _stats(outcome_any))
    _fmt_block("Time to T1 (for trades that touched T1)", _stats(t1_any))
    _fmt_block("Time to T2 (for T2 winners only)", _stats(t2_only))
    print("-----------------------------------")

def movers_detailed_table(rds: RedisState, max_rows: int = 100):
    book = rds.load_json("state", "silent_open", default={})

    movers = [
        tr for tr in book.values()
        if tr.get("source") == "movers" and tr.get("status") == "closed"
    ]

    if not movers:
        print("--- Movers Detailed Table: no closed movers trades yet ---")
        return

    def _parse_ts(x):
        try:
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT

    movers.sort(key=lambda tr: _parse_ts(tr.get("opened_at")))

    if max_rows is not None and max_rows > 0:
        movers = movers[-max_rows:]

    print(f"--- Movers Detailed Trades (latest {len(movers)}) ---")
    header = (
        "opened_at".ljust(20)
        + " closed_at".ljust(20)
        + " symbol".ljust(14)
        + " tf".ljust(4)
        + " entry".rjust(12)
        + " stop".rjust(12)
        + " t1".rjust(12)
        + " t2".rjust(12)
        + " outcome".ljust(10)
        + " exit".rjust(12)
        + "  R".rjust(8)
        + " t_out(min)".rjust(12)
        + " t_t1(min)".rjust(12)
        + " t_t2(min)".rjust(12)
    )
    print(header)
    print("-" * len(header))

    def _fmt(x, width=12, prec=6):
        if x is None:
            return " " * width
        try:
            return f"{float(x):{width}.{prec}f}"
        except Exception:
            s = str(x)
            return s[:width].rjust(width)

    for tr in movers:
        opened = (tr.get("opened_at") or "")[:19]
        closed = (tr.get("closed_at") or "")[:19]
        sym = tr.get("symbol", "")[:13]
        tf = tr.get("tf", "")

        entry = _fmt(tr.get("entry"))
        stop = _fmt(tr.get("stop"))
        t1 = _fmt(tr.get("t1"))
        t2 = _fmt(tr.get("t2"))
        exit_p = _fmt(tr.get("exit_price"))
        R = _fmt(tr.get("R"), width=8, prec=2)

        t_out = _fmt(tr.get("time_to_outcome_min"), width=12, prec=1)
        t_t1 = _fmt(tr.get("time_to_t1_min"), width=12, prec=1)
        t_t2 = _fmt(tr.get("time_to_t2_min"), width=12, prec=1)

        outcome = (tr.get("outcome") or "")[:9]

        line = (
            opened.ljust(20)
            + " " + closed.ljust(20)
            + " " + sym.ljust(14)
            + " " + str(tf).ljust(4)
            + entry
            + stop
            + t1
            + t2
            + " " + outcome.ljust(10)
            + exit_p
            + R
            + t_out
            + t_t1
            + t_t2
        )
        print(line)

    print("-" * len(header))

# =============== Fallback universe (optional) ===============
def mexc_top_usdt_volume_pairs(client: ExClient, *, max_pairs=60, min_usd_vol=2_000_000, extra_stables=None):
    extra_stables = extra_stables or []
    try:
        tickers = client.ex.fetch_tickers()
    except Exception as e:
        print("[fallback] fetch_tickers err:", e)
        return []
    items = []
    for sym, t in tickers.items():
        if "/USDT" not in sym:
            continue
        if is_stable_or_pegged(sym, extra_stables):
            continue
        qv = t.get("quoteVolume")
        if qv is None:
            base_v = t.get("baseVolume")
            last = t.get("last") or t.get("close")
            try:
                qv = (base_v or 0) * (last or 0)
            except Exception:
                qv = 0
        try:
            qv = float(qv or 0)
        except Exception:
            qv = 0.0
        if qv >= float(min_usd_vol):
            items.append((sym, qv))
    items.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s, _ in items[:max_pairs]]
    print(f"[fallback] MEXC top USDT-volume picked {len(pairs)} pairs (min_usd_vol={min_usd_vol})")
    return pairs

# =============== Discord helpers ===============
def _post_discord(hook: str, text: str):
    if not hook or not text.strip():
        return
    try:
        resp = requests.post(hook, json={"content": text}, timeout=10)
        if resp.status_code >= 300:
            print(f"[discord] post err: HTTP {resp.status_code} {resp.text}")
    except Exception as e:
        print("[discord] post err:", e)

def fmt_mover_signal_block(sig: Dict[str, Any]) -> str:
    symbol = sig["symbol"]
    tf = sig["timeframe"]
    note = sig.get("note", "")
    try:
        entry = float(sig["entry"])
    except Exception:
        entry = 0.0

    # raw levels
    stop_raw = sig.get("stop")
    t1_raw = sig.get("t1")
    t2_raw = sig.get("t2")

    stop = float(stop_raw) if stop_raw is not None else None
    t1 = float(t1_raw) if t1_raw is not None else None
    t2 = float(t2_raw) if t2_raw is not None else None

    # Percent calculations (safe)
    risk_pct = None
    t1_pct = None
    t2_pct = None

    if entry > 0 and stop is not None:
        try:
            risk_pct = (entry - stop) / entry * 100.0
        except ZeroDivisionError:
            risk_pct = None

    if entry > 0 and t1 is not None and t1 > entry:
        try:
            t1_pct = (t1 / entry - 1.0) * 100.0
        except ZeroDivisionError:
            t1_pct = None

    if entry > 0 and t2 is not None and t2 > entry:
        try:
            t2_pct = (t2 / entry - 1.0) * 100.0
        except ZeroDivisionError:
            t2_pct = None

    def pct(x):
        return f"{x:.1f}%" if x is not None else "n/a"

    direction = "Long"  # Movers are long breakout signals

    header = "**Kritocurrency Alpha Signals – Movers Signal 🚀**"

    lines = [
        header,
        "",
        f"> **`{symbol}` — {tf} {note} ({direction})**",
        f"> • Entry: `{entry:.6f}`",
    ]

    if stop is not None:
        lines.append(f"> • Stop-loss: `{stop:.6f}`  (~{pct(risk_pct)} risk)")

    if t1 is not None:
        lines.append(f"> • Target 1: `{t1:.6f}`  (~{pct(t1_pct)} from entry)")

    if t2 is not None:
        lines.append(f"> • Target 2: `{t2:.6f}`  (~{pct(t2_pct)} from entry)")

    lines.append("")
    lines.append("_Use position sizing and your own risk management. Not financial advice._")

    return "\n".join(lines)

# =============== RUN ===============
def run(cfg: Dict[str, Any]):
    client = ExClient()
    rds = RedisState(
        url=(cfg.get("persistence", {}).get("redis_url") or os.environ.get("REDIS_URL")),
        prefix=cfg.get("persistence", {}).get("key_prefix", "spideybot:v1"),
        ttl_minutes=int(cfg.get("persistence", {}).get("ttl_minutes", 2880)),
    )

    # Discord hook
    SIG_HOOK = os.environ.get("DISCORD_SIGNALS_WEBHOOK") or cfg.get("discord", {}).get("signals_webhook", "")

    # Filters
    extra_stables = cfg.get("filters", {}).get("extra_stables", [])

    # Reporting
    reporting_cfg = cfg.get("reporting", {})
    movers_max_rows = int(reporting_cfg.get("movers_detailed_max_rows", 100))

    # Movers config
    mv_cfg = cfg.get("movers", {"enabled": True})
    fb_cfg = cfg.get("fallback", {"enabled": False})

    # Universe — Movers
    movers_syms = cmc_movers_symbols(cfg) if mv_cfg.get("enabled", True) else []
    movers_pairs = filter_pairs_on_mexc(client, movers_syms, mv_cfg.get("quote", "USDT"), extra_stables)
    print(f"[universe] Movers mapped -> {len(movers_pairs)} pairs")

    # Optional fallback
    if fb_cfg.get("enabled", False) and len(movers_pairs) == 0:
        movers_pairs = mexc_top_usdt_volume_pairs(
            client,
            max_pairs=int(fb_cfg.get("max_pairs", 80)),
            min_usd_vol=float(fb_cfg.get("min_usd_vol", 2_000_000)),
            extra_stables=extra_stables,
        )
        print("[universe] Movers fallback -> using MEXC volume universe")

    results_movers: List[Dict[str, Any]] = []

    # ======== Movers Scan ========
    for pair in movers_pairs:
        try:
            df1h = client.ohlcv(pair, "1h", 260)
            sig = legacy_mover_signal(df1h)
            if sig:
                sig.update({"symbol": pair, "timeframe": "1h", "exchange": "mexc"})
                if not is_silent_open_movers(rds, pair, "1h", sig["type"]):
                    results_movers.append(sig)
                    record_new_silent(rds, sig, "movers")
        except Exception as e:
            print(f"[scan-movers] {pair} err:", e)

    # ======== Evaluate/close silent movers ========
    close_silent_if_hit(rds, client, cfg)
    movers_stats_table(rds)
    movers_time_stats_table(rds)
    movers_detailed_table(rds, movers_max_rows)

    # ======== Logs ========
    print(f"=== Movers scan done @ {pd.Timestamp.utcnow().isoformat()} ===")
    print(f"Scanned Movers pairs: {len(movers_pairs)}")

    # ======== Discord (Movers only) ========
    for sig in results_movers:
        msg = fmt_mover_signal_block(sig)
        _post_discord(SIG_HOOK, msg)

# =============== Entrypoint ===============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_movers_bot_config.yml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    def expand_env(o):
        if isinstance(o, dict):
            return {k: expand_env(v) for k, v in o.items()}
        if isinstance(o, list):
            return [expand_env(x) for x in o]
        if isinstance(o, str) and o.startswith("${") and o.endswith("}"):
            return os.environ.get(o[2:-1], o)
        return o

    cfg = expand_env(cfg)
    run(cfg)
