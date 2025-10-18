#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_sd_scanner.py — Render Cron scanner with multi-exchange support
- EXCHANGES env var: comma-separated list (e.g., "mexc,gate,binance")
- Separate lists: symbols_day (1h), symbols_trend (1d)
- Optional SD zones; optional CMC "movers" intake; optional Telegram
- Signals include the exchange name; logs + JSON printed for Render
"""
import argparse, json, os, yaml, sys, time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import ccxt

# ---------------- math helpers ----------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(length).mean()
    down = -delta.clip(upper=0).rolling(length).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ---------------- exchange wrapper ----------------
class ExClient:
    def __init__(self, name: str):
        klass = getattr(ccxt, name)
        self.name = name
        self.ex = klass({"enableRateLimit": True})
        self._markets = None
    def load_markets(self):
        if self._markets is None:
            try:
                self._markets = self.ex.load_markets()
            except Exception:
                self._markets = {}
        return self._markets
    def has_pair(self, symbol_pair: str) -> bool:
        mkts = self.load_markets()
        return (symbol_pair in mkts) or (symbol_pair in getattr(self.ex, "symbols", []) or False)
    def ohlcv(self, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

# ---------------- params ----------------
@dataclass
class DayParams:
    lookback_high: int = 20
    vol_sma: int = 20
    rsi_min: int = 50
    rsi_max: int = 80
    btc_filter: bool = True
    btc_symbol: str = "BTC/USDT"
    btc_ema: int = 20
    stop_mode: str = "swing"
    atr_mult: float = 1.5
@dataclass
class TrendParams:
    ema20: int = 20
    ema50: int = 50
    ema100: int = 100
    pullback_pct_max: float = 12.0
    rsi_min: int = 50
    rsi_max: int = 70
    vol_sma: int = 20
    breakout_lookback: int = 55
    stop_mode: str = "swing"
    atr_mult: float = 2.0

# ---------------- SD zones ----------------
def body(df): return (df['close'] - df['open']).abs()
def avg_range(df, n=20): return (df['high'] - df['low']).rolling(n).mean()
def find_zones(df: pd.DataFrame, impulse_factor=1.8, zone_padding_pct=0.25, max_age_bars=300):
    zones = []
    ar = avg_range(df, 20)
    b = body(df)
    for i in range(20, len(df)):
        ref = ar.iloc[i] if not np.isnan(ar.iloc[i]) else 0
        if b.iloc[i] > impulse_factor * ref:
            bullish = df['close'].iloc[i] > df['open'].iloc[i]
            j = i - 1
            if bullish and df['close'].iloc[j] < df['open'].iloc[j]:
                lo, hi = df['low'].iloc[j], df['high'].iloc[j]
                pad = (hi - lo) * zone_padding_pct
                zones.append({"type":"demand","low":float(lo-pad),"high":float(hi+pad),"index":i-1})
            if (not bullish) and df['close'].iloc[j] > df['open'].iloc[j]:
                lo, hi = df['low'].iloc[j], df['high'].iloc[j]
                pad = (hi - lo) * zone_padding_pct
                zones.append({"type":"supply","low":float(lo-pad),"high":float(hi+pad),"index":i-1})
    return [z for z in zones if (len(df) - z["index"]) <= max_age_bars]
def in_demand(price: float, zones) -> bool:
    for z in zones or []:
        if z["type"]=="demand" and z["low"]<=price<=z["high"]: return True
    return False

# ---------------- signals ----------------
def stop_from(df, mode, atr_mult):
    if mode=="atr":
        a=atr(df,14).iloc[-1]; a=0 if np.isnan(a) else float(a)
        return float(df['close'].iloc[-1] - atr_mult*a)
    return float(min(df['low'].iloc[-10:]))
def day_signal(df, p: DayParams, sd_cfg, zones=None):
    look=p.lookback_high; voln=p.vol_sma
    if len(df)<max(look,voln)+5: return None
    volS=sma(df['volume'], voln); r=rsi(df['close'],14); last=df.iloc[-1]
    highlvl=df['high'].iloc[-(look+1):-1].max()
    breakout=(last['close']>highlvl) and (last['volume']>volS.iloc[-1]) and (p.rsi_min<=r.iloc[-1]<=p.rsi_max)
    retrec=(df['low'].iloc[-1]<=highlvl) and (last['close']>highlvl) and (last['volume']>0.8*volS.iloc[-1])
    if not (breakout or retrec): return None
    if sd_cfg.get('enabled'):
        if sd_cfg.get('mode','prefer')=='require' and not in_demand(float(last['close']), zones):
            return None
    entry=float(last['close']); stop=stop_from(df, p.stop_mode, p.atr_mult)
    sig={'type':'day','entry':entry,'stop':stop,'t1':round(entry*1.05,6),'t2':round(entry*1.10,6),
         'level':float(highlvl),'note':'Breakout' if breakout else 'Retest-Reclaim'}
    if sd_cfg.get('enabled'): sig['sd_confluence']=in_demand(entry, zones)
    return sig
def trend_signal(df, p: TrendParams, sd_cfg, zones=None):
    need=max(p.ema100, p.vol_sma, p.breakout_lookback)+5
    if len(df)<need: return None
    df=df.copy()
    df['ema20']=ema(df['close'], p.ema20); df['ema50']=ema(df['close'], p.ema50); df['ema100']=ema(df['close'], p.ema100)
    df['volS']=sma(df['volume'], p.vol_sma); r=rsi(df['close'],14); last=df.iloc[-1]
    aligned=(last['ema20']>last['ema50']>last['ema100']) and (r.iloc[-1]>=p.rsi_min)
    within=abs((last['close']-last['ema20'])/last['ema20']*100)<=p.pullback_pct_max
    bounce=last['close']>df['close'].iloc[-2]
    highlvl=df['high'].iloc[-(p.breakout_lookback+1):-1].max()
    breakout=(last['close']>highlvl) and (last['volume']>df['volS'].iloc[-1])
    if not (aligned and ((within and bounce) or breakout)): return None
    if sd_cfg.get('enabled'):
        if sd_cfg.get('mode','prefer')=='require' and not in_demand(float(last['close']), zones):
            return None
    entry=float(last['close']); stop=stop_from(df, p.stop_mode, p.atr_mult)
    sig={'type':'trend','entry':entry,'stop':stop,'t1':round(entry*1.08,6),'t2':round(entry*1.20,6),
         'level':float(highlvl),'note':'Pullback-Bounce' if (within and bounce) else 'Breakout',
         'ema20':float(last['ema20']),'ema50':float(last['ema50']),'ema100':float(last['ema100'])}
    if sd_cfg.get('enabled'): sig['sd_confluence']=in_demand(entry, zones)
    return sig

# ---------------- BTC filter ----------------
def btc_ok(ex: ExClient, dayp: DayParams) -> bool:
    try:
        df=ex.ohlcv(dayp.btc_symbol,'1h',120)
        return df['close'].iloc[-1] > ema(df['close'], dayp.btc_ema).iloc[-1]
    except Exception: return True

# ---------------- Movers via CMC ----------------
def fetch_cmc_list(cfg: Dict[str,Any]) -> List[str]:
    mv=cfg.get('movers',{})
    if not mv.get('enabled'): return []
    api_key=os.environ.get('CMC_API_KEY') or mv.get('cmc_api_key')
    if not api_key: 
        print('[movers] enabled but no CMC_API_KEY — skipping'); return []
    limit=mv.get('limit',200); min_change=mv.get('min_change_24h',15.0)
    min_vol=mv.get('min_volume_usd_24h',2_000_000); max_age_days=mv.get('max_age_days',365)
    include=[s.strip().upper() for s in mv.get('include_symbols','').split(',') if s.strip()]
    exclude=set([s.strip().upper() for s in mv.get('exclude_symbols','').split(',') if s.strip()])
    import requests
    url='https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    params={'limit':limit,'convert':'USD','sort':'percent_change_24h','sort_dir':'desc'}
    headers={'X-CMC_PRO_API_KEY': api_key}
    try:
        r=requests.get(url, params=params, headers=headers, timeout=15)
        data=r.json()['data']
    except Exception as e:
        print('[movers] error:', e); return []
    now=pd.Timestamp.utcnow(); out=[]
    for it in data:
        sym=it['symbol'].upper()
        if sym in exclude: continue
        if include and (sym in include): out.append(sym); continue
        ch=it.get('quote',{}).get('USD',{}).get('percent_change_24h',0) or 0
        vol=it.get('quote',{}).get('USD',{}).get('volume_24h',0) or 0
        date_added=pd.to_datetime(it.get('date_added', now.isoformat()), utc=True)
        age=(now-date_added).days
        if (ch>=min_change) and (vol>=min_vol) and (age<=max_age_days):
            out.append(sym)
    return out

def filter_pairs_on_exchanges(ex_clients: Dict[str,ExClient], symbols: List[str], quote='USDT') -> Dict[str,List[str]]:
    # returns dict: ex_name -> [ "SYM/USDT", ... ]
    by_ex={name:[] for name in ex_clients.keys()}
    for name, client in ex_clients.items():
        for sym in symbols:
            pair=f"{sym}/{quote}"
            if client.has_pair(pair):
                by_ex[name].append(pair)
    return by_ex

# ---------------- utils ----------------
def parse_csv_env(val):
    if isinstance(val, list): return val
    if isinstance(val, str): return [s.strip() for s in val.split(',') if s.strip()]
    return []

# ---------------- run ----------------
def run(cfg: Dict[str,Any]):
    # exchanges
    ex_names = parse_csv_env(cfg.get('exchanges') or 'mexc')
    ex_clients = {name: ExClient(name) for name in ex_names}

    # groups
    syms_day = parse_csv_env(cfg.get('symbols_day', ''))
    syms_trn = parse_csv_env(cfg.get('symbols_trend', ''))
    # params
    dayP = DayParams(**cfg.get('day_trade_params', {}))
    trnP = TrendParams(**cfg.get('trend_trade_params', {}))
    sd_cfg = cfg.get('supply_demand', {'enabled': False})
    mv_cfg = cfg.get('movers', {'enabled': False})

    # movers list (symbols only), then per-exchange tradables
    movers_pairs_by_ex: Dict[str,List[str]] = {name:[] for name in ex_clients.keys()}
    if mv_cfg.get('enabled'):
        cmc_syms = fetch_cmc_list(cfg)
        by_ex = filter_pairs_on_exchanges(ex_clients, cmc_syms, mv_cfg.get('quote','USDT'))
        movers_pairs_by_ex = by_ex

    # zones cache per exchange+pair
    zones_cache: Dict[Tuple[str,str], List[Dict[str,Any]]] = {}
    if sd_cfg.get('enabled'):
        ztf=sd_cfg.get('timeframe_for_zones','1h'); zlook=sd_cfg.get('lookback',300)
        for ex_name, client in ex_clients.items():
            # set of all pairs we might scan on this exchange
            pairs = set(syms_day + syms_trn + movers_pairs_by_ex.get(ex_name,[]))
            for pair in pairs:
                if not client.has_pair(pair): 
                    continue
                try:
                    zdf = client.ohlcv(pair, ztf, zlook)
                    zones_cache[(ex_name,pair)] = find_zones(zdf, sd_cfg.get('impulse_factor',1.8),
                                                             sd_cfg.get('zone_padding_pct',0.25),
                                                             sd_cfg.get('max_age_bars',300))
                except Exception as e:
                    print(f"[zones] {ex_name} {pair} err:", e)
                    zones_cache[(ex_name,pair)] = []

    results = {"ts": pd.Timestamp.utcnow().isoformat(), "signals": [], "movers": movers_pairs_by_ex}

    # scan per exchange
    for ex_name, client in ex_clients.items():
        # Day list = manual list + movers for this exchange
        day_pairs = [p for p in syms_day if client.has_pair(p)]
        day_pairs = list(dict.fromkeys(day_pairs + movers_pairs_by_ex.get(ex_name, [])))

        # BTC filter for this exchange context
        allow_day = (not dayP.btc_filter) or btc_ok(client, dayP)

        for pair in day_pairs:
            try:
                df1h = client.ohlcv(pair, '1h', 300)
                zones = zones_cache.get((ex_name,pair))
                sig = day_signal(df1h, dayP, sd_cfg, zones)
                if allow_day and sig:
                    sig.update({"symbol": pair, "timeframe": "1h", "exchange": ex_name})
                    results["signals"].append(sig)
            except Exception as e:
                print(f"[scan-day] {ex_name} {pair} err:", e)

        # Trend list = manual only for now (mov ers are for fast day-trade lists)
        trend_pairs = [p for p in syms_trn if client.has_pair(p)]
        for pair in trend_pairs:
            try:
                dfd = client.ohlcv(pair, '1d', 300)
                zones = zones_cache.get((ex_name,pair))
                sig = trend_signal(dfd, trnP, sd_cfg, zones)
                if sig:
                    sig.update({"symbol": pair, "timeframe": "1d", "exchange": ex_name})
                    results["signals"].append(sig)
            except Exception as e:
                print(f"[scan-trend] {ex_name} {pair} err:", e)

    # logs
    print(f"=== Crypto Signals @ {results['ts']} ===")
    for ex_name, pairs in results["movers"].items():
        if pairs:
            print(f"Movers considered on {ex_name}: {', '.join(pairs)}")
    if results["signals"]:
        for s in results["signals"]:
            sd_tag = " ✅SD" if s.get("sd_confluence") else ""
            print(f"[{s['exchange']}] {s['symbol']} {s['timeframe']} {s['type'].upper()} — {s['note']}{sd_tag} — "
                  f"entry {s['entry']} stop {s['stop']} t1 {s['t1']} t2 {s['t2']}")
    else:
        print("No signals this run.")
    print(json.dumps(results, indent=2))

    # optional telegram
    tcfg = cfg.get("telegram", {"enabled": False})
    if tcfg.get("enabled") and results["signals"]:
        try:
            import requests
            lines = ["*Crypto Signals*"]
            for s in results["signals"]:
                sd_tag = " ✅SD" if s.get("sd_confluence") else ""
                lines.append(f"• [{s['exchange']}] `{s['symbol']}` *{s['type'].upper()}* {s['timeframe']} — {s['note']}{sd_tag}\n"
                             f"  entry `{s['entry']}` stop `{s['stop']}` t1 `{s['t1']}` t2 `{s['t2']}`")
            requests.post(f"https://api.telegram.org/bot{tcfg['bot_token']}/sendMessage",
                          json={"chat_id": tcfg["chat_id"], "text":"\n".join(lines), "parse_mode":"Markdown"},
                          timeout=10)
        except Exception as e:
            print("[telegram] err:", e)

    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Expand env-style placeholders for top-level fields
    def env_or(val, default=None):
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            return os.environ.get(val[2:-1], default)
        return val
    for k in ["exchanges", "symbols_day", "symbols_trend"]:
        if k in cfg:
            cfg[k] = env_or(cfg[k], cfg[k])

    run(cfg)
