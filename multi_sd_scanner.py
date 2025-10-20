exchanges: ["mexc", "gate", "binance"]
symbols_watchlist: ["SOL/USDT", "BNB/USDT", "ETH/USDT", "COAI/USDT", "MNT/USDT"]

quality:
  min_confidence: 70
  signal_cooldown_bars: 6
  require_two_closes_breakout: false
  min_avg_dollar_vol_1h: 200000
  edge_entries: true

day_trade_params:
  lookback_high: 20
  vol_sma: 30
  rsi_min: 52
  rsi_max: 78
  btc_filter: true
  btc_symbol: "BTC/USDT"
  btc_ema: 20
  stop_mode: "swing"   # or "atr"
  atr_mult: 1.5
  early_reversal:
    enabled: true
    require_ema_cross: true
    require_macd_cross: true
    min_rsi: 55
    max_extension_pct: 6
    min_vol_mult: 1.0
  multi_tf:
    enabled: true
    confirm_tfs: ["5m","15m","30m"]
    min_confirmations: 2
    require_ema_stack: true
    require_macd_bull: true
    min_rsi: 52

swing_trade_params:
  timeframe: "4h"
  ema20: 20
  ema50: 50
  ema100: 100
  pullback_pct_max: 10.0
  rsi_min: 50
  vol_sma: 20
  breakout_lookback: 34
  stop_mode: "swing"
  atr_mult: 1.8

trend_trade_params:
  ema20: 20
  ema50: 50
  ema100: 100
  pullback_pct_max: 10.0
  rsi_min: 50
  rsi_max: 70
  vol_sma: 20
  breakout_lookback: 55
  stop_mode: "swing"
  atr_mult: 2.0

supply_demand:
  enabled: false
  mode: "prefer"                # or "require"
  timeframe_for_zones: "1h"
  lookback: 300
  impulse_factor: 1.8
  zone_padding_pct: 0.25
  max_age_bars: 300

movers:
  enabled: true
  limit: 500
  quote: "USDT"
  min_change_24h: 15.0
  min_volume_usd_24h: 5000000
  max_age_days: 365
  # env: CMC_API_KEY

exits:
  enabled: true
  state_file: "state.json"
  edge_trigger: true
  day:
    ema_break: 20
    rsi_drop_from: 70
    rsi_drop_to: 60
    macd_confirm: true
    multi_tf:
      enabled: true
      confirm_tfs: ["5m","15m","30m"]
      min_confirmations: 2
      require_ema_bear: true
      require_macd_bear: true
      max_rsi: 50
  swing:
    ema_break: 50
    rsi_below: 50
    macd_confirm: true
  trend:
    rsi_below: 50
    ema_cross_20_50: true
    macd_confirm: true

bearish_signals:
  enabled: false
  day:
    timeframe: "1h"
    lookback_low: 20
    vol_sma: 20
    require_breakdown: true
    require_vol_confirm: true
    sd:
      require_supply: false
    multi_tf:
      enabled: true
      confirm_tfs: ["5m","15m","30m"]
      min_confirmations: 2
      require_ema_bear: true
      require_macd_bear: true
      max_rsi: 50
  swing:
    timeframe: "4h"
    lookback_low: 34
    ema_stack_bear: true
    require_vol_confirm: true
    rsi_max: 50
  trend:
    timeframe: "1d"
    lookback_low: 55
    ema20_below_50: true
    rsi_max: 50

performance:
  enabled: true
  csv_path: "perf_trades.csv"
  assume_fills: "next_close"   # "signal_close" or "next_close"
  tp_priority: "target_first"  # or "stop_first"
  use_exit_signals: true
  max_bars_eval:
    day: 96     # ~4d on 1h
    swing: 180  # ~30d on 4h
    trend: 365  # ~1y on 1d

telegram:
  enabled: false
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"

discord:
  enabled: false
  webhook: https://discord.com/api/webhooks/1429125113780899981/8jqvbb4Idij4GFWvFqaCGPdT2NBDnloP_aoleyY4Czxop0Gvfmv8kqQN019ynqmFb1y2
