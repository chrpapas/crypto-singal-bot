#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, yaml, requests, time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np
import ccxt

# ================== TA helpers ==================
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff(); up = d.clip(lower=0).rolling(length).mean(); dn = -d.clip(upper=0).rolling(length).mean()
    rs = up / (dn + 1e-9); return 100 - (100 / (1 + rs))
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df['high'] - df['low']; hc = (df['high'] - df['close'].shift()).abs(); lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl,hc,lc], axis=1).max(axis=1); return tr.rolling(n).mean()
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = ema(series, fast); s = ema(series, slow); line = f - s; sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig; return line, sig, hist

# ================== Exchange wrapper ==================
class ExClient:
    def __init__(self, name: str):
        self.name = name
        self.ex = getattr(ccxt, name)({"enableRateLimit": True})
        self._markets = None
    def load_markets(self):
        if self._markets is None:
            try: self._markets = self.ex.load_markets()
            except Exception: self._markets = {}
        return self._markets
    def has_pair(self, symbol_pair: str) -> bool:
        mkts = self.load_markets() or {}
        if symbol_pair in mkts: return True
        syms = getattr(self.ex, "symbols", None) or []
        return symbol_pair in syms
    def ohlcv(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

# ================== Params ==================
@dataclass
class DayParams:
    lookback_high: int = 30
    vol_sma: int = 30
    rsi_min: int = 52
    rsi_max: int = 78
    btc_filter: bool = True
    btc_symbol: str = "BTC/USDT"
    btc_ema: int = 20
    stop_mode: str = "swing"
    atr_mult: float = 1.5
    early_reversal: dict = field(default_factory=dict)
    multi_tf: dict = field(default_factory=dict)

@dataclass
class TrendParams:
    ema20: int = 20
    ema50: int = 50
    ema100: int = 100
    pullback_pct_max: float = 10.0
    rsi_min: int = 50
    rsi_max: int = 70
    vol_sma: int = 20
    breakout_lookback: int = 55
    stop_mode: str = "swing"
    atr_mult: float = 2.0

# ================== SD zones ==================
def body(df: pd.DataFrame) -> pd.Series: return (df['close'] - df['open']).abs()
def avg_range(df: pd.DataFrame, n: int = 20) -> pd.Series: return (df['high'] - df['low']).rolling(n).mean()
def find_zones(df: pd.DataFrame, impulse_factor=1.8, zone_padding_pct=0.25, max_age_bars=300):
    zones = []; ar = avg_range(df, 20); b = body(df)
    for i in range(20, len(df)):
        ref = ar.iloc[i] if not np.isnan(ar.iloc[i]) else 0
        if b.iloc[i] > impulse_factor * ref:
            bull = df['close'].iloc[i] > df['open'].iloc[i]; j = i-1
            lo,hi = df['low'].iloc[j], df['high'].iloc[j]; pad = (hi-lo)*zone_padding_pct
            if bull and df['close'].iloc[j] < df['open'].iloc[j]:
                zones.append({"type":"demand","low":float(lo-pad),"high":float(hi+pad),"index":i-1})
            if (not bull) and df['close'].iloc[j] > df['open'].iloc[j]:
                zones.append({"type":"supply","low":float(lo-pad),"high":float(hi+pad),"index":i-1})
    return [z for z in zones if (len(df) - z["index"]) <= max_age_bars]
def in_demand(price: float, zones) -> bool:
    for z in zones or []:
        if z["type"]=="demand" and z["low"]<=price<=z["high"]: return True
    return False

# ================== Stops ==================
def stop_from(df: pd.DataFrame, mode: str, atr_mult: float) -> float:
    if mode == "atr":
        a = atr(df, 14).iloc[-1]; a = 0 if np.isnan(a) else float(a)
        return float(df["close"].iloc[-1] - atr_mult * a)
    return float(min(df["low"].iloc[-10:]))

# ================== EDGE helpers ==================
def _cross_down(a_now, a_prev, b_now, b_prev) -> bool:
    return (a_prev >= b_prev) and (a_now < b_now)
def _cross_up(a_now, a_prev, b_now, b_prev) -> bool:
    return (a_prev <= b_prev) and (a_now > b_now)

# ================== DAY entries (edge) ==================
def day_signal(df: pd.DataFrame, p: DayParams, sd_cfg: Dict[str,Any], zones=None):
    look, voln = p.lookback_high, p.vol_sma
    if len(df) < max(look, voln)+5: return None
    volS = sma(df["volume"], voln); r = rsi(df["close"], 14)
    last, prev = df.iloc[-1], df.iloc[-2]
    highlvl = df["high"].iloc[-(look+1):-1].max()

    breakout_edge = (prev["close"] <= highlvl) and (last["close"] > highlvl)
    breakout_ok   = breakout_edge and (last["volume"]>volS.iloc[-1]) and (p.rsi_min<=r.iloc[-1]<=p.rsi_max)

    retest_edge = (prev["low"] <= highlvl) and (prev["close"] <= highlvl) and (last["close"] > highlvl)
    retrec_ok   = retest_edge and (last["volume"]>0.8*volS.iloc[-1]) and (r.iloc[-1]>=p.rsi_min)

    if not (breakout_ok or retrec_ok): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None

    entry = float(last["close"]); stop = stop_from(df, p.stop_mode, p.atr_mult)
    sig = {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.05,6),"t2":round(entry*1.10,6),
           "level":float(highlvl),"note":"Breakout" if breakout_ok else "Retest-Reclaim",
           "event_bar_ts": df.index[-1].isoformat()}
    if sd_cfg.get("enabled"): sig["sd_confluence"] = in_demand(entry, zones)
    return sig

def day_early_reversal_signal(df: pd.DataFrame, p_map: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    er = p_map.get("early_reversal", {}) or {}
    if not er.get("enabled", False): return None
    if len(df) < max(60, p_map.get("vol_sma",20)) + 5: return None
    e20  = ema(df["close"], 20); e50  = ema(df["close"], 50)
    volS = sma(df["volume"], p_map.get("vol_sma", 20)); r = rsi(df["close"], 14)
    macd_line, macd_sig, _ = macd(df["close"])
    last_close = df["close"].iloc[-1]; last_vol = df["volume"].iloc[-1]

    prev_cross_ok = (e20.iloc[-2] <= e50.iloc[-2]) if er.get("require_ema_cross", True) else True
    ema_cross = (e20.iloc[-1] > e50.iloc[-1]) and prev_cross_ok
    macd_ok = (macd_line.iloc[-1] > macd_sig.iloc[-1]) if er.get("require_macd_cross", True) else True
    rsi_ok = r.iloc[-1] >= er.get("min_rsi", 55)
    ext = abs((last_close - e20.iloc[-1]) / (e20.iloc[-1] + 1e-9) * 100.0); not_extended = ext <= er.get("max_extension_pct", 6)
    vol_ok = last_vol >= er.get("min_vol_mult", 1.0) * (volS.iloc[-1] if not np.isnan(volS.iloc[-1]) else last_vol)
    ok = all([ema_cross, macd_ok, rsi_ok, not_extended, vol_ok])
    if not ok: return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer") == "require" and not in_demand(float(last_close), zones): return None

    entry = float(last_close); stop  = stop_from(df, p_map.get("stop_mode","swing"), p_map.get("atr_mult",1.5))
    sig = {"type":"day","entry":entry,"stop":stop,"t1":round(entry*1.04,6),"t2":round(entry*1.08,6),
           "level": float(e20.iloc[-1]), "note":"Early-Reversal (EMA20>50 + MACD + RSI)",
           "event_bar_ts": df.index[-1].isoformat()}
    if sd_cfg.get("enabled"): sig["sd_confluence"] = in_demand(entry, zones)
    return sig

# ================== SWING & TREND entries ==================
def swing_signal(df: pd.DataFrame, p: Dict[str,Any], sd_cfg: Dict[str,Any], zones=None):
    need = max(p.get('ema100',100), p.get('vol_sma',20), p.get('breakout_lookback',34))+5
    if len(df)<need: return None
    df=df.copy(); df['ema20']=ema(df['close'],p.get('ema20',20)); df['ema50']=ema(df['close'],p.get('ema50',50))
    df['ema100']=ema(df['close'],p.get('ema100',100)); df['volS']=sma(df['volume'],p.get('vol_sma',20))
    r=rsi(df['close'],14); last=df.iloc[-1]
    aligned=(last['ema20']>last['ema50']>last['ema100']) and (r.iloc[-1]>=p.get('rsi_min',50))
    within=abs((last['close']-last['ema20'])/last['ema20']*100)<=p.get('pullback_pct_max',10.0)
    bounce=last['close']>df['close'].iloc[-2]
    hl=df['high'].iloc[-(p.get('breakout_lookback',34)+1):-1].max()
    breakout=(last['close']>hl) and (last['volume']>df['volS'].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get('enabled') and sd_cfg.get('mode','prefer')=='require' and not in_demand(float(last['close']), zones): return None
    entry=float(last['close']); stop=stop_from(df,p.get('stop_mode','swing'),p.get('atr_mult',1.8))
    return {"type":"swing","entry":entry,"stop":stop,"t1":round(entry*1.06,6),"t2":round(entry*1.12,6),
            "level":float(hl),"note":"4h Pullback-Bounce" if (within and bounce) else "4h Breakout",
            "ema20":float(last['ema20']),"ema50":float(last['ema50']),"ema100":float(last['ema100'])}

def trend_signal(df: pd.DataFrame, p: TrendParams, sd_cfg: Dict[str,Any], zones=None):
    need=max(p.ema100,p.vol_sma,p.breakout_lookback)+5
    if len(df)<need: return None
    df=df.copy(); df["ema20"]=ema(df["close"],p.ema20); df["ema50"]=ema(df["close"],p.ema50)
    df["ema100"]=ema(df["close"],p.ema100); df["volS"]=sma(df["volume"],p.vol_sma)
    r=rsi(df["close"],14); last=df.iloc[-1]
    aligned=(last["ema20"]>last["ema50"]>last["ema100"]) and (r.iloc[-1]>=p.rsi_min)
    within=abs((last["close"]-last["ema20"])/last["ema20"]*100)<=p.pullback_pct_max
    bounce=last["close"]>df["close"].iloc[-2]
    hl=df["high"].iloc[-(p.breakout_lookback+1):-1].max()
    breakout=(last["close"]>hl) and (last["volume"]>df["volS"].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get("enabled") and sd_cfg.get("mode","prefer")=="require" and not in_demand(float(last["close"]), zones): return None
    entry=float(last["close"]); stop=stop_from(df,p.stop_mode,p.atr_mult)
    sig={"type":"trend","entry":entry,"stop":stop,"t1":round(entry*1.08,6),"t2":round(entry*1.20,6),
         "level":float(hl),"note":"Pullback-Bounce" if (within and bounce) else "Breakout",
         "ema20":float(df['ema20'].iloc[-1]),"ema50":float(df['ema50'].iloc[-1]),"ema100":float(df['ema100'].iloc[-1])}
    if sd_cfg.get("enabled"): sig["sd_confluence"]=in_demand(entry, zones)
    return sig

# ================== BTC market filter ==================
def btc_ok(ex: ExClient, dayp: DayParams) -> bool:
    try:
        df = ex.ohlcv(dayp.btc_symbol, "1h", 120)
        return df["close"].iloc[-1] > ema(df["close"], dayp.btc_ema).iloc[-1]
    except Exception:
        return True

# ================== Exit helpers (edge + non-edge) ==================
def _macd_bear(df):
    line, sig, _ = macd(df["close"])
    return line.iloc[-1] < sig.iloc[-1], (line.iloc[-2] >= sig.iloc[-2]) and (line.iloc[-1] < sig.iloc[-1])

def day_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 60: return False, "", None
    p_ema = int(cfg.get("ema_break", 20)); p_from = int(cfg.get("rsi_drop_from", 70)); p_to = int(cfg.get("rsi_drop_to", 60))
    macd_need = bool(cfg.get("macd_confirm", True))
    e = ema(df["close"], p_ema); r = rsi(df["close"], 14)
    close_now, close_prev = df["close"].iloc[-1], df["close"].iloc[-2]; e_now, e_prev = e.iloc[-1], e.iloc[-2]
    cross_down = (close_prev >= e_prev) and (close_now < e_now)
    if not cross_down: return False, "", df.index[-1]
    rsi_drop_ok = (r.iloc[-2] >= p_from) and (r.iloc[-1] <= p_to)
    macd_bear, macd_cross = _macd_bear(df); macd_ok = macd_bear if not macd_need else macd_cross
    if rsi_drop_ok or macd_ok:
        why = f"EMA{p_ema} cross-down; " + ("RSI drop" if rsi_drop_ok else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def swing_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 120: return False, "", None
    p_ema = int(cfg.get("ema_break", 50)); rsi_below = int(cfg.get("rsi_below", 50)); macd_need = bool(cfg.get("macd_confirm", True))
    e = ema(df["close"], p_ema); r = rsi(df["close"], 14)
    close_now, close_prev = df["close"].iloc[-1], df["close"].iloc[-2]; e_now, e_prev = e.iloc[-1], e.iloc[-2]
    cross_down = (close_prev >= e_prev) and (close_now < e_now)
    macd_bear, macd_cross = _macd_bear(df); macd_ok = macd_cross if macd_need else macd_bear
    if cross_down and (r.iloc[-1] <= rsi_below or macd_ok):
        why = f"EMA{p_ema} cross-down; " + ("RSI<=%d" % rsi_below if r.iloc[-1] <= rsi_below else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

def trend_exit_edge(df: pd.DataFrame, cfg: Dict[str,Any]):
    if df is None or len(df) < 200: return False, "", None
    need_cross = bool(cfg.get("ema_cross_20_50", True)); rsi_below = int(cfg.get("rsi_below", 50)); macd_need = bool(cfg.get("macd_confirm", True))
    e20, e50 = ema(df["close"],20), ema(df["close"],50)
    cross20_50 = (e20.iloc[-2] >= e50.iloc[-2]) and (e20.iloc[-1] < e50.iloc[-1]) if need_cross else True
    r = rsi(df["close"], 14); macd_bear, macd_cross = _macd_bear(df); macd_ok = macd_cross if macd_need else macd_bear
    if cross20_50 and (r.iloc[-1] <= rsi_below or macd_ok):
        why = ("EMA20<EMA50 cross; " if need_cross else "") + ("RSI<=%d" % rsi_below if r.iloc[-1] <= rsi_below else "MACD bear")
        return True, why, df.index[-1]
    return False, "", df.index[-1]

# ================== Main run ==================
def run(cfg: Dict[str,Any]):
    ex_names = parse_csv_env(cfg.get("exchanges") or os.environ.get("EXCHANGES","mexc"))
    ex_clients = {name: ExClient(name) for name in ex_names}
    watchlist = parse_csv_env(cfg.get("symbols_watchlist") or os.environ.get("SYMBOLS_WATCHLIST","BTC/USDT,SOL/USDT"))

    dayP = DayParams(**cfg.get("day_trade_params", {}))
    swingP = cfg.get("swing_trade_params", {"timeframe":"4h"})
    trnP = TrendParams(**cfg.get("trend_trade_params", {}))
    sd_cfg = cfg.get("supply_demand", {"enabled": False})
    mv_cfg = cfg.get("movers", {"enabled": False})
    exits_cfg = cfg.get("exits", {"enabled": True})
    pfcfg = cfg.get("performance", {"enabled": True})
    state_path = exits_cfg.get("state_file","state.json")
    state = load_state(state_path)

    results = {"ts": pd.Timestamp.utcnow().isoformat(), "signals": [], "exit_signals": [], "movers": {}}

    for ex_name, client in ex_clients.items():
        base_pairs = [p for p in watchlist if client.has_pair(p)]
        # DAY
        if (not dayP.btc_filter) or btc_ok(client, dayP):
            for pair in base_pairs:
                try:
                    df1h = client.ohlcv(pair, "1h", 300)
                    sig = day_signal(df1h, dayP, sd_cfg, None) or day_early_reversal_signal(df1h, vars(dayP), sd_cfg, None)
                    if sig:
                        sig.update({"symbol":pair,"timeframe":"1h","exchange":ex_name})
                        results["signals"].append(sig)
                        add_open_trade(state, exchange=ex_name, symbol=pair, tf="1h", sig_type="day",
                                       entry=sig["entry"], stop=sig["stop"], t1=sig.get("t1"), t2=sig.get("t2"),
                                       event_ts=sig.get("event_bar_ts", pd.Timestamp.utcnow().isoformat()))
                except Exception as e: print(f"[scan-day] {ex_name} {pair} err:", e)
        # SWING
        for pair in base_pairs:
            try:
                df4h = client.ohlcv(pair, "4h", 400)
                sig = swing_signal(df4h, swingP, sd_cfg, None)
                if sig:
                    sig.update({"symbol":pair,"timeframe":"4h","exchange":ex_name})
                    results["signals"].append(sig)
                    add_open_trade(state, exchange=ex_name, symbol=pair, tf="4h", sig_type="swing",
                                   entry=sig["entry"], stop=sig["stop"], t1=sig.get("t1"), t2=sig.get("t2"),
                                   event_ts=df4h.index[-1].isoformat())
            except Exception as e: print(f"[scan-swing] {ex_name} {pair} err:", e)
        # TREND
        for pair in base_pairs:
            try:
                dfd = client.ohlcv(pair, "1d", 320)
                sig = trend_signal(dfd, trnP, sd_cfg, None)
                if sig:
                    sig.update({"symbol":pair,"timeframe":"1d","exchange":ex_name})
                    results["signals"].append(sig)
                    add_open_trade(state, exchange=ex_name, symbol=pair, tf="1d", sig_type="trend",
                                   entry=sig["entry"], stop=sig["stop"], t1=sig.get("t1"), t2=sig.get("t2"),
                                   event_ts=dfd.index[-1].isoformat())
            except Exception as e: print(f"[scan-trend] {ex_name} {pair} err:", e)

        # EXITS
        for pair in base_pairs:
            try:
                df1h = client.ohlcv(pair,"1h",200)
                ok, why, bar_ts = day_exit_edge(df1h, exits_cfg.get("day", {}))
                if ok:
                    results["exit_signals"].append({"type":"day_exit","reason":why,"symbol":pair,"timeframe":"1h","exchange":ex_name})
                    remove_position(state, ex_name, pair, "day")
                df4h = client.ohlcv(pair,"4h",300)
                ok, why, bar_ts = swing_exit_edge(df4h, exits_cfg.get("swing", {}))
                if ok:
                    results["exit_signals"].append({"type":"swing_exit","reason":why,"symbol":pair,"timeframe":"4h","exchange":ex_name})
                    remove_position(state, ex_name, pair, "swing")
                dfd = client.ohlcv(pair,"1d",320)
                ok, why, bar_ts = trend_exit_edge(dfd, exits_cfg.get("trend", {}))
                if ok:
                    results["exit_signals"].append({"type":"trend_exit","reason":why,"symbol":pair,"timeframe":"1d","exchange":ex_name})
                    remove_position(state, ex_name, pair, "trend")
            except Exception as e: print(f"[exits] {ex_name} {pair} err:", e)

    if pfcfg.get("enabled", True):
        evaluate_open_trades(state, ex_clients, cfg)
        write_perf_csv(state, pfcfg.get("csv_path","perf_trades.csv"))
    save_state(state_path, state)

    # Logs
    print(f"=== Crypto Signals @ {results['ts']} ===")
    if results["signals"]:
        for s in results["signals"]:
            print(f"[ENTRY] [{s['exchange']}] {s['symbol']} {s['timeframe']} {s['type'].upper()} — {s['note']} — entry {s['entry']} stop {s['stop']} t1 {s['t1']} t2 {s['t2']}")
    else:
        print("No entry signals this run.")
    if results["exit_signals"]:
        for x in results["exit_signals"]:
            print(f"[EXIT ] [{x['exchange']}] {x['symbol']} {x['timeframe']} — {x['type']} — {x['reason']}")
    else:
        print("No exit signals this run.")

    # Discord (optional)
    dcfg = cfg.get("discord", {"enabled": False})
    if dcfg.get("enabled") and (results["signals"] or results["exit_signals"]):
        try:
            lines=[]
            if results["signals"]:
                lines.append("**Entries**")
                for s in results["signals"]:
                    lines.append(f"**[{s['exchange']}] {s['symbol']}** ({s['timeframe']} {s['type'].upper()}) — {s['note']}\nentry `{s['entry']}` stop `{s['stop']}` t1 `{s['t1']}` t2 `{s['t2']}`")
            if results["exit_signals"]:
                lines.append("**Exits**")
                for x in results["exit_signals"]:
                    lines.append(f"**[{x['exchange']}] {x['symbol']}** ({x['timeframe']}) — {x['type']} — {x['reason']}")
            requests.post(dcfg["webhook"], json={"content":"\n".join(lines)}, timeout=10)
        except Exception as e: print("[discord] err:", e)

# ================== Entrypoint ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    if os.path.exists(args.config):
        with open(args.config,"r") as f: cfg = yaml.safe_load(f)
    else:
        cfg = {}
    # expand ${VAR}
    def expand_env(obj):
        if isinstance(obj, dict): return {k: expand_env(v) for k,v in obj.items()}
        if isinstance(obj, list): return [expand_env(x) for x in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], obj)
        return obj
    cfg = expand_env(cfg)
    run(cfg)