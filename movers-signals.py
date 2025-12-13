#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, yaml, requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

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

# =============== Redis persistence ===============
def _normalize_redis_url(url: str) -> str:
    if not url or not isinstance(url, str):
        raise RuntimeError("Redis URL missing in config (persistence.redis_url).")

    u = url.strip().strip('"').strip("'")
    if u.startswith("http://"):
        u = "redis://" + u[len("http://"):]
    if u.startswith("https://"):
        u = "rediss://" + u[len("https://"):]
    if not (u.startswith("redis://") or u.startswith("rediss://") or u.startswith("unix://")):
        raise RuntimeError(f"Invalid Redis URL scheme: {u.split(':',1)[0]} (need redis:// or rediss:// or unix://)")
    return u

class RedisState:
    def __init__(self, url: str, prefix: str, ttl_minutes: int):
        url = _normalize_redis_url(url)
        self.prefix = prefix or "spideybot:v1"
        self.ttl_seconds = int(ttl_minutes) * 60 if ttl_minutes else 48 * 3600
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=6)

        # only this selftest key uses TTL
        self.r.setex(self.k("selftest"), self.ttl_seconds, pd.Timestamp.utcnow().isoformat())
        print(f"[persistence] Redis OK | prefix={self.prefix} | edge-ttl-min={self.ttl_seconds//60}")

    def k(self, *parts) -> str:
        return ":".join([self.prefix, *[str(p) for p in parts]])

    def get(self, *parts) -> Optional[str]:
        return self.r.get(self.k(*parts))

    def set(self, val: str, *parts):
        self.r.set(self.k(*parts), val)

    def load_json(self, *parts, default=None):
        txt = self.get(*parts)
        if not txt:
            return default if default is not None else {}
        try:
            return json.loads(txt)
        except Exception:
            return default if default is not None else {}

    def save_json(self, obj, *parts):
        # IMPORTANT: no TTL here -> history persists
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
    headers = {"X-CMC_PRO_API_KEY": cfg.get("movers", {}).get("cmc_api_key", "")}
    if not headers["X-CMC_PRO_API_KEY"] or str(headers["X-CMC_PRO_API_KEY"]).startswith("${"):
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
    data = fetch_cmc_listings(cfg, limit=int(mv.get("limit", 500)))
    out = []
    now = pd.Timestamp.utcnow()
    min_change = float(mv.get("min_change_24h", 8.0))
    min_vol = float(mv.get("min_volume_usd_24h", 5_000_000))
    max_age = int(mv.get("max_age_days", 3650))

    for it in data:
        sym = it.get("symbol", "").upper()
        q = it.get("quote", {}).get("USD", {})
        ch = float(q.get("percent_change_24h") or 0.0)
        vol = float(q.get("volume_24h") or 0.0)
        date_added = pd.to_datetime(it.get("date_added", now.isoformat()), utc=True)
        age_days = (now - date_added).days
        if (ch >= min_change) and (vol >= min_vol) and (age_days <= max_age):
            out.append(sym)
    return out

def filter_pairs_on_mexc(client: ExClient, symbols: List[str], quote: str, extra_stables: List[str]) -> List[str]:
    client.load_markets()
    out = []
    for sym in symbols:
        pair = f"{sym}/{quote}"
        if client.has_pair(pair) and not is_stable_or_pegged(pair, extra_stables):
            out.append(pair)
    return out

# =============== Signal Logic ===============
def legacy_mover_signal(df1h: pd.DataFrame, mv_cfg: Dict[str, Any], dbg: Dict[str,int]) -> Optional[Dict[str, Any]]:
    if df1h is None or len(df1h) < 80:
        dbg["too_short"] += 1
        return None

    risk_cfg = mv_cfg.get("risk_filter", {}) or {}
    min_risk_pct = float(risk_cfg.get("min_risk_pct", 0.8))
    max_risk_pct = float(risk_cfg.get("max_risk_pct", 14.0))
    min_rel_volume = float(mv_cfg.get("min_rel_volume", 1.0))

    rsi_min = float(mv_cfg.get("rsi_min", 52))
    rsi_max = float(mv_cfg.get("rsi_max", 82))

    e20, e50 = ema(df1h["close"], 20), ema(df1h["close"], 50)
    mac_line, mac_sig, _ = macd(df1h["close"])
    r = rsi(df1h["close"], 14)
    vS = sma(df1h["volume"], 20)

    last = df1h.iloc[-1]
    hl = df1h["high"].iloc[-31:-1].max()

    aligned = (e20.iloc[-1] > e50.iloc[-1] and mac_line.iloc[-1] > mac_sig.iloc[-1] and rsi_min <= r.iloc[-1] <= rsi_max)
    if not aligned:
        dbg["reject_aligned"] += 1
        return None

    avg_vol = vS.iloc[-1] if not np.isnan(vS.iloc[-1]) else float(last["volume"])
    if float(last["volume"]) < min_rel_volume * float(avg_vol):
        dbg["reject_volume"] += 1
        return None

    if not (float(last["close"]) > float(hl)):
        dbg["reject_breakout"] += 1
        return None

    entry = float(last["close"])
    stop = float(min(df1h["low"].iloc[-10:]))

    if entry <= 0 or stop <= 0 or stop >= entry:
        dbg["reject_bad_levels"] += 1
        return None

    risk_pct = (entry - stop) / entry * 100.0
    if risk_pct < min_risk_pct:
        dbg["reject_risk_too_tight"] += 1
        return None
    if risk_pct > max_risk_pct:
        dbg["reject_risk_too_wide"] += 1
        return None

    dbg["signal"] += 1
    return {
        "type": "day",
        "entry": entry,
        "stop": stop,
        "t1": round(entry * 1.05, 6),
        "t2": round(entry * 1.10, 6),
        "level": float(hl),
        "note": "Mover Trend",
        "event_bar_ts": df1h.index[-1].isoformat(),
    }

# =============== Silent registry ===============
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
    }
    rds.save_json(book, "state", "silent_open")

def is_silent_open_movers(rds: RedisState, symbol: str, tf: str, typ: str) -> bool:
    book = rds.load_json("state", "silent_open", default={})
    return silent_key(symbol, tf, typ, "movers") in book

def close_silent_if_hit(rds: RedisState, client: ExClient, mv_cfg: Dict[str, Any]):
    book = rds.load_json("state", "silent_open", default={})
    if not book:
        return

    timeout_hours = float(mv_cfg.get("open_timeout_hours", 168))
    now_utc = datetime.now(timezone.utc)

    changed = False

    for k, tr in list(book.items()):
        if tr.get("status") != "open":
            continue

        pair = tr["symbol"]
        tf = tr["tf"]
        stop = tr.get("stop")
        t1 = tr.get("t1")
        t2 = tr.get("t2")

        opened_at = pd.to_datetime(tr["opened_at"], utc=True)

        # --- timeout close (prevents “open forever”) ---
        age_hours = (now_utc - opened_at.to_pydatetime()).total_seconds() / 3600.0
        if age_hours >= timeout_hours:
            tr["status"] = "closed"
            tr["closed_at"] = pd.Timestamp(now_utc).isoformat()
            tr["outcome"] = "timeout"
            tr["exit_price"] = None
            tr["R"] = None
            book[k] = tr
            changed = True
            continue

        try:
            df = client.ohlcv(pair, tf, 400 if tf in ("1h","4h") else 330)
            df = df.loc[opened_at:]
            rows = df.iloc[1:]

            outcome = None
            exit_price = None
            hit_ts = None

            for ts, r in rows.iterrows():
                lo, hi = float(r["low"]), float(r["high"])
                touched_stop = (stop is not None and lo <= stop)
                touched_t1   = (t1 is not None and hi >= t1)
                touched_t2   = (t2 is not None and hi >= t2)

                if not (touched_stop or touched_t1 or touched_t2):
                    continue

                if touched_t2:
                    outcome, exit_price, hit_ts = "t2", t2, ts
                elif touched_t1:
                    outcome, exit_price, hit_ts = "t1", t1, ts
                else:
                    outcome, exit_price, hit_ts = "stop", stop, ts
                break

            if outcome:
                tr["status"] = "closed"
                tr["closed_at"] = hit_ts.isoformat()
                tr["outcome"] = outcome
                tr["exit_price"] = float(exit_price)

                try:
                    entry = float(tr.get("entry"))
                    stop_val = float(tr.get("stop")) if tr.get("stop") is not None else None
                    if stop_val is not None:
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

# =============== Fallback universe ===============
def mexc_top_usdt_volume_pairs(client: ExClient, *, max_pairs=120, min_usd_vol=2_000_000, extra_stables=None):
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
            qv = float(base_v or 0) * float(last or 0)
        qv = float(qv or 0)
        if qv >= float(min_usd_vol):
            items.append((sym, qv))
    items.sort(key=lambda x: x[1], reverse=True)
    pairs = [s for s, _ in items[:max_pairs]]
    print(f"[fallback] MEXC top USDT-volume picked {len(pairs)} pairs (min_usd_vol={min_usd_vol})")
    return pairs

# =============== Discord ===============
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
    entry = float(sig["entry"])

    stop = float(sig["stop"]) if sig.get("stop") is not None else None
    t1 = float(sig["t1"]) if sig.get("t1") is not None else None
    t2 = float(sig["t2"]) if sig.get("t2") is not None else None

    def pct(x): return f"{x:.1f}%" if x is not None else "n/a"
    risk_pct = ((entry - stop) / entry * 100.0) if (stop is not None and entry > 0) else None
    t1_pct = ((t1 / entry - 1.0) * 100.0) if (t1 is not None and entry > 0) else None
    t2_pct = ((t2 / entry - 1.0) * 100.0) if (t2 is not None and entry > 0) else None

    lines = [
        "**Kritocurrency Alpha Signals – Movers Signal 🚀**",
        "",
        f"> **`{symbol}` — {tf} {note} (Long)**",
        f"> • Entry: `{entry:.6f}`",
    ]
    if stop is not None: lines.append(f"> • Stop-loss: `{stop:.6f}`  (~{pct(risk_pct)} risk)")
    if t1 is not None: lines.append(f"> • Target 1: `{t1:.6f}`  (~{pct(t1_pct)} from entry)")
    if t2 is not None: lines.append(f"> • Target 2: `{t2:.6f}`  (~{pct(t2_pct)} from entry)")
    lines += ["", "_Use position sizing and your own risk management. Not financial advice._"]
    return "\n".join(lines)

# =============== RUN ===============
def run(cfg: Dict[str, Any]):
    client = ExClient()

    rds = RedisState(
        url=cfg.get("persistence", {}).get("redis_url"),
        prefix=cfg.get("persistence", {}).get("key_prefix", "spideybot:v1"),
        ttl_minutes=int(cfg.get("persistence", {}).get("ttl_minutes", 2880)),
    )

    SIG_HOOK = cfg.get("discord", {}).get("signals_webhook", "")
    extra_stables = cfg.get("filters", {}).get("extra_stables", [])

    mv_cfg = cfg.get("movers", {"enabled": True})
    fb_cfg = cfg.get("fallback", {"enabled": False})

    movers_syms = cmc_movers_symbols(cfg) if mv_cfg.get("enabled", True) else []
    movers_pairs = filter_pairs_on_mexc(client, movers_syms, mv_cfg.get("quote", "USDT"), extra_stables)
    print(f"[universe] Movers mapped -> {len(movers_pairs)} pairs")

    min_universe = int(mv_cfg.get("min_universe_pairs", 0))
    if fb_cfg.get("enabled", False) and (len(movers_pairs) == 0 or (min_universe and len(movers_pairs) < min_universe)):
        movers_pairs = mexc_top_usdt_volume_pairs(
            client,
            max_pairs=int(fb_cfg.get("max_pairs", 120)),
            min_usd_vol=float(fb_cfg.get("min_usd_vol", 2_000_000)),
            extra_stables=extra_stables,
        )
        print("[universe] Using fallback volume universe")

    # evaluate closes + timeouts first (reduces “open forever” blocking)
    close_silent_if_hit(rds, client, mv_cfg)

    dbg = {
        "too_short": 0, "reject_aligned": 0, "reject_breakout": 0, "reject_volume": 0,
        "reject_bad_levels": 0, "reject_risk_too_tight": 0, "reject_risk_too_wide": 0,
        "signal": 0,
    }

    results_movers: List[Dict[str, Any]] = []
    for pair in movers_pairs:
        try:
            df1h = client.ohlcv(pair, "1h", 260)
            sig = legacy_mover_signal(df1h, mv_cfg=mv_cfg, dbg=dbg)
            if sig:
                sig.update({"symbol": pair, "timeframe": "1h", "exchange": "mexc"})
                if not is_silent_open_movers(rds, pair, "1h", sig["type"]):
                    results_movers.append(sig)
                    record_new_silent(rds, sig, "movers")
        except Exception as e:
            print(f"[scan-movers] {pair} err:", e)

    print(f"[debug] signal filters summary: {dbg}")
    print(f"=== Movers scan done @ {pd.Timestamp.utcnow().isoformat()} ===")
    print(f"Scanned Movers pairs: {len(movers_pairs)}")
    print(f"New signals emitted : {len(results_movers)}")

    for sig in results_movers:
        _post_discord(SIG_HOOK, fmt_mover_signal_block(sig))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mexc_movers_bot_config.yml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # expand ${VARS} except you asked not to use env for Redis (we don’t)
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
