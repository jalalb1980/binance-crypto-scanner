import asyncio
import aiohttp
import numpy as np
import random
import time
from datetime import datetime
from math import isfinite

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'

MAX_CONCURRENT_REQUESTS = 30        # tuned to avoid 418s and finish faster
HTTP_TIMEOUT = 15
RETRY_ATTEMPTS = 3
CANDLE_LIMIT = 160                  # enough for indicators; faster than 200

# Thresholds
MIN_SCORE_EARLY = 3
MIN_SCORE_CONFIRMED = 4
PRICE_CHANGE_THRESHOLD = 2.0        # +/-2% for confirmed
VOLUME_SPIKE_RATIO = 2.0

TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
PRICE_TF = HIGH_TF
TRIANGLE_TF = MID_TF

# --- StochRSI crossover section (separate list) ---
# Default stays on higher timeframe (1d). To switch to 30m, set STOCH_TF = LOW_TF
STOCH_TF = HIGH_TF
STOCH_BEAR_MIN_DROP = 2.0  # Bearish Stoch list must have price change <= -2%
STOCH_BULL_MIN_RISE = 2.0  # Bullish Stoch list must have price change >= +2%

# Strategy toggles (tags only; do not affect filtering)
USE_LTR = True
USE_TBB = True
USE_VSQ = True
USE_AVWAP = True

# --- Anti-418 / failover settings ---
BINANCE_FAPI_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi.binancefuture.com",
    "https://fapi1.binance.com",
]
SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/118.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

# Simple in-process cache for symbols
_SYMBOL_CACHE = None
_SYMBOL_CACHE_TTL_SEC = 6 * 60 * 60  # 6 hours
_SYMBOL_CACHE_AT = None

EPS = 1e-12

# === HELPERS ===
def is_futures_usdt(sym):
    return sym.get('quoteAsset') == 'USDT' and sym.get('contractType') == 'PERPETUAL' and sym.get('status') == 'TRADING'

async def fetch_json(session, url, method="GET", data=None):
    """Failover across hosts + respectful backoff for 418/429/403/451."""
    last_exc = None
    for attempt in range(6):
        base = random.choice(BINANCE_FAPI_HOSTS)
        for h in BINANCE_FAPI_HOSTS:
            if url.startswith(h):
                url = url.replace(h, base)
                break
        try:
            kwargs = {"timeout": HTTP_TIMEOUT, "headers": SESSION_HEADERS}
            req = session.post if method == "POST" else session.get
            if method == "POST":
                kwargs["data"] = data
            async with req(url, **kwargs) as res:
                if res.status in (418, 429, 451, 403, 409):
                    retry_after = res.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else 1.0 + min(8, attempt) + random.random()
                    await asyncio.sleep(wait)
                    last_exc = f"HTTP {res.status}"
                    continue
                if res.status == 200:
                    return await res.json()
                last_exc = f"HTTP {res.status}"
        except Exception as e:
            last_exc = e
        await asyncio.sleep(0.5 * (2 ** (attempt // 2)) + random.random() * 0.5)
    raise RuntimeError(f"GET failed: {url} -> {last_exc}")

async def fetch_symbols(session):
    """Try exchangeInfo first; if blocked, fall back to ticker/price. Cache for 6h."""
    global _SYMBOL_CACHE, _SYMBOL_CACHE_AT
    now = time.time()
    if _SYMBOL_CACHE and _SYMBOL_CACHE_AT and (now - _SYMBOL_CACHE_AT) < _SYMBOL_CACHE_TTL_SEC:
        return _SYMBOL_CACHE
    for base in BINANCE_FAPI_HOSTS:
        try:
            data = await fetch_json(session, f"{base}/fapi/v1/exchangeInfo")
            syms = [s['symbol'] for s in data['symbols'] if is_futures_usdt(s)]
            if syms:
                _SYMBOL_CACHE = syms
                _SYMBOL_CACHE_AT = now
                return syms
        except Exception:
            continue
    for base in BINANCE_FAPI_HOSTS:
        try:
            tick = await fetch_json(session, f"{base}/fapi/v1/ticker/price")
            syms = sorted({t['symbol'] for t in tick if t['symbol'].endswith('USDT')})
            _SYMBOL_CACHE = syms
            _SYMBOL_CACHE_AT = now
            return syms
        except Exception:
            continue
    raise RuntimeError("Could not fetch symbols (blocked/ratelimited). Try again later.")

async def fetch_candles(session, symbol, interval, limit=CANDLE_LIMIT):
    base = random.choice(BINANCE_FAPI_HOSTS)
    url = f"{base}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    return await fetch_json(session, url)

def to_ohlcv(candles):
    o = np.array([float(x[1]) for x in candles], dtype=float)
    h = np.array([float(x[2]) for x in candles], dtype=float)
    l = np.array([float(x[3]) for x in candles], dtype=float)
    c = np.array([float(x[4]) for x in candles], dtype=float)
    v = np.array([float(x[5]) for x in candles], dtype=float)
    return o, h, l, c, v

# === INDICATORS (robust, no external libs) ===
def ema(arr, period):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0 or period <= 0:
        return np.array([])
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def rsi_wilder(closes, period=14):
    c = np.asarray(closes, dtype=float)
    if len(c) < period + 1:
        return np.array([])
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.empty(len(delta)); avg_loss = np.empty(len(delta))
    avg_gain[:period] = np.nan; avg_loss[:period] = np.nan
    avg_gain[period-1] = gain[:period].mean()
    avg_loss[period-1] = loss[:period].mean()
    for i in range(period, len(delta)):
        avg_gain[i] = (avg_gain[i-1]*(period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1]*(period-1) + loss[i]) / period
    rs = avg_gain / np.maximum(avg_loss, EPS)
    rs[~np.isfinite(rs)] = 0.0
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.concatenate(([np.nan], rsi))
    return rsi

def macd_histogram(closes, fast=12, slow=26, signal=9):
    c = np.asarray(closes, dtype=float)
    if len(c) < slow + signal + 5:
        return np.nan
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return float(hist[-1])

def bollinger_bands_signal(closes, period=20, dev=2.0):
    c = np.asarray(closes, dtype=float)
    if len(c) < period:
        return False
    ma = c[-period:].mean()
    sd = c[-period:].std(ddof=0)
    upper = ma + dev*sd
    lower = ma - dev*sd
    return bool(c[-1] > upper or c[-1] < lower)

def psar_signal(highs, lows, closes, af_step=0.02, af_max=0.2):
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if n < 5:  # fallback
        return c[-1] > c[-2]
    uptrend = c[1] > c[0]
    ep = h[0] if uptrend else l[0]
    af = af_step
    psar = c[0]
    for i in range(1, n):
        prev_psar = psar
        if uptrend:
            psar = prev_psar + af * (ep - prev_psar)
            psar = min(psar, l[i-1], l[i-2] if i >= 2 else l[i-1])
            if h[i] > ep:
                ep = h[i]; af = min(af + af_step, af_max)
            if l[i] < psar:
                uptrend = False; psar = ep; ep = l[i]; af = af_step
        else:
            psar = prev_psar + af * (ep - prev_psar)
            psar = max(psar, h[i-1], h[i-2] if i >= 2 else h[i-1])
            if l[i] < ep:
                ep = l[i]; af = min(af + af_step, af_max)
            if h[i] > psar:
                uptrend = True; psar = ep; ep = h[i]; af = af_step
    return bool(c[-1] > psar)

def stoch_rsi_signal(closes, period=14):
    # True StochRSI (K,D in 0..100)
    r = rsi_wilder(closes, period)
    valid = r[~np.isnan(r)]
    if len(valid) < period + 3:
        return None
    rsi_window = valid[-period:]
    rmin = np.min(rsi_window); rmax = np.max(rsi_window)
    denom = max(rmax - rmin, EPS)
    k = (valid[-1] - rmin) / denom * 100.0
    k_series = []
    for back in range(3, 0, -1):
        win = valid[-(period + (3 - back)) : -(3 - back) if (3 - back) != 0 else None]
        if len(win) < period:
            k_series.append(k); continue
        rmin_b = np.min(win[-period:]); rmax_b = np.max(win[-period:])
        denom_b = max(rmax_b - rmin_b, EPS)
        k_series.append((win[-1] - rmin_b) / denom_b * 100.0)
    d = float(np.mean(k_series))
    diff = abs(k - d)
    signal = None
    if k > d and k < 40:
        signal = 'bullish'
    elif k < d and k > 60:
        signal = 'bearish'
    level = 'Hot' if diff <= 3 else 'Good' if diff <= 15 else 'Normal'
    return signal, diff, level, k, d

# === PATTERN / STRATEGY TAGS (informational only) ===
def detect_triangle_tag(closes, highs, lows, lookback=30, tight=0.02):
    if len(closes) < lookback:
        return '(Tx)'
    hb = float(np.max(highs[-lookback:])); lb = float(np.min(lows[-lookback:]))
    width = hb - lb
    if width <= 0:
        return '(Tx)'
    if width / max(closes[-1], EPS) < tight:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return '(Tx)'

def tag_liquidity_trap_reversal(closes, highs, lows, volumes):
    if not USE_LTR or len(closes) < 6:
        return None
    range_hl = highs[-1] - lows[-1]
    if range_hl <= 0:
        return None
    vol_spike = volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * 2.0 if len(volumes) > 20 else False
    tail = closes[-1] - lows[-1]
    if (tail / max(range_hl, EPS) > 0.6) and vol_spike:
        return 'LTR▲'
    upper_wick = highs[-1] - closes[-1]
    if (upper_wick / max(range_hl, EPS) > 0.6) and vol_spike:
        return 'LTR▼'
    return None

def tag_time_based_breakout(closes, opens, volumes, timestamps=None, compress_len=20, width_pct=0.01):
    if not USE_TBB or len(closes) < compress_len + 5:
        return None
    window = closes[-compress_len:]
    w = (np.max(window) - np.min(window)) / max(window[-1], EPS)
    if w < width_pct:
        prev_max = np.max(window[:-1]); prev_min = np.min(window[:-1])
        if closes[-1] > prev_max and volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * 1.5:
            return 'TBB▲'
        if closes[-1] < prev_min and volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * 1.5:
            return 'TBB▼'
    return None

def tag_volatility_squeeze(closes, period=20, dev=2.0, pct_thresh=0.08):
    if not USE_VSQ or len(closes) < period + 5:
        return None
    c = np.asarray(closes, dtype=float)
    ma = c[-period:].mean()
    sd = c[-period:].std(ddof=0)
    width = (2 * dev * sd) / max(ma, EPS)
    if width < pct_thresh:
        if c[-1] > ma + dev*sd: return 'VSQ▲'
        if c[-1] < ma - dev*sd: return 'VSQ▼'
    return None

def tag_avwap_reclaim(closes, highs, lows, anchor_lookback=60):
    if not USE_AVWAP or len(closes) < anchor_lookback + 5:
        return None
    swing_low_idx = np.argmin(lows[-anchor_lookback:])
    swing_high_idx = np.argmax(highs[-anchor_lookback:])
    def avwap_from(idx):
        start = len(closes) - anchor_lookback + idx
        c = np.array(closes[start:], dtype=float)
        v = np.ones_like(c)
        return float(np.sum(c * v) / np.sum(v))
    avwap_low = avwap_from(swing_low_idx)
    avwap_high = avwap_from(swing_high_idx)
    if closes[-1] > avwap_low and closes[-2] <= avwap_low: return 'AVWAP▲'
    if closes[-1] < avwap_high and closes[-2] >= avwap_high: return 'AVWAP▼'
    return None

# === ACTION HINT ===
def derive_action(overall_trend, label, strength, momentum_score):
    """
    Returns one of: LONG (A), LONG (B), SHORT (A), SHORT (B)
    """
    if overall_trend == 'bullish':
        if label == "(Confirmed)":
            return "LONG (A)" if strength == "Strong" else "LONG (B)"
        else:
            return "LONG (B)"  # Early = B-tier probe
    else:  # bearish
        if label == "(Confirmed)":
            return "SHORT (A)" if strength == "Strong" else "SHORT (B)"
        else:
            return "SHORT (B)"

# === ANALYSIS ===
async def analyze_stoch(session, symbol, semaphore):
    async with semaphore:
        try:
            candles = await fetch_candles(session, symbol, STOCH_TF)
            o, h, l, c, v = to_ohlcv(candles)
            sig = stoch_rsi_signal(c)
            if not sig:
                return None
            signal, diff, level, k, d = sig
            pc = (c[-1] - c[-2]) / max(c[-2], EPS) * 100.0
            label = f"**{symbol}** | {pc:+.2f}% | {signal.upper()} | K:{k:.1f} D:{d:.1f} | Δ:{diff:.2f} | 🔥 {level}"
            return signal, pc, label
        except Exception:
            return None

def indicator_bundle(highs, lows, closes):
    ema_fast_up = ema(closes, 9)
    ema_slow_up = ema(closes, 21)
    ema_ok = bool(ema_fast_up.size and ema_slow_up.size and ema_fast_up[-1] > ema_slow_up[-1])

    rsi_val_arr = rsi_wilder(closes, 14)
    rsi_ok = bool(rsi_val_arr.size and isfinite(rsi_val_arr[-1]) and rsi_val_arr[-1] > 50)

    mh = macd_histogram(closes)
    macd_ok = bool(isfinite(mh) and mh > 0)

    sar_ok = psar_signal(highs, lows, closes)
    boll_ok = bollinger_bands_signal(closes)

    stoch = stoch_rsi_signal(closes)
    stoch_sig = stoch[0] if stoch else None
    stoch_ok = (stoch_sig == 'bullish')

    return {'EMA': ema_ok,'RSI': rsi_ok,'MACD': macd_ok,'SAR': sar_ok,'BOLL': boll_ok,'STOCH': stoch_ok}

def get_trend_direction(closes):
    e9 = ema(closes, 9); e21 = ema(closes, 21)
    if len(e9) == 0 or len(e21) == 0: return 'neutral'
    if e9[-1] > e21[-1]: return 'bullish'
    if e9[-1] < e21[-1]: return 'bearish'
    return 'neutral'

def detect_momentum(closes, volumes):
    e9 = ema(closes, 9); e21 = ema(closes, 21)
    rsi_arr = rsi_wilder(closes, 14)
    macd_h = macd_histogram(closes)
    if len(e9) < 1 or len(e21) < 1 or len(rsi_arr) < 1:
        return 0, False
    vol_spike = False
    if len(volumes) > 20:
        vol_spike = volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * VOLUME_SPIKE_RATIO
    score = 0
    if e9[-1] > e21[-1]: score += 1
    if rsi_arr[-1] > 50: score += 1
    if isfinite(macd_h) and macd_h > 0: score += 1
    if vol_spike: score += 1
    return score, vol_spike

async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            # Fetch needed TF candles concurrently
            tfs = list(set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF]))
            tasks = [fetch_candles(session, symbol, tf) for tf in tfs]
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            candles = {}
            for tf, data in zip(tfs, fetched):
                if isinstance(data, Exception):
                    return None
                candles[tf] = data

            # Build arrays
            series = {}
            for tf in tfs:
                o,h,l,c,v = to_ohlcv(candles[tf])
                series[tf] = {'o':o,'h':h,'l':l,'c':c,'v':v}

            last_price = float(series[PRICE_TF]['c'][-1])
            price_change = (series[PRICE_TF]['c'][-1] - series[PRICE_TF]['c'][-2]) / max(series[PRICE_TF]['c'][-2], EPS) * 100.0

            # Indicators across MID/HIGH
            indicators_summary = {}
            for tf in INDICATOR_TFS:
                ind = indicator_bundle(series[tf]['h'], series[tf]['l'], series[tf]['c'])
                for k, v in ind.items():
                    indicators_summary[k] = indicators_summary.get(k, 0) + (1 if v else 0)
            indicator_score = sum(1 for v in indicators_summary.values() if v > 0)

            # Momentum on MOMENTUM_TF (aligned volumes)
            momentum_score, vol_spike = detect_momentum(series[MOMENTUM_TF]['c'], series[MOMENTUM_TF]['v'])

            # Triangle/icon + Tags (on TRIANGLE_TF arrays)
            tri = detect_triangle_tag(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['h'], series[TRIANGLE_TF]['l'])
            tags = []
            t1 = tag_liquidity_trap_reversal(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['h'], series[TRIANGLE_TF]['l'], series[TRIANGLE_TF]['v']);  t1 and tags.append(t1)
            t2 = tag_time_based_breakout(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['o'], series[TRIANGLE_TF]['v']);  t2 and tags.append(t2)
            t3 = tag_volatility_squeeze(series[TRIANGLE_TF]['c']);                                                       t3 and tags.append(t3)
            t4 = tag_avwap_reclaim(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['h'], series[TRIANGLE_TF]['l']);       t4 and tags.append(t4)
            tag_str = (" [" + ",".join(tags) + "]") if tags else ""

            # Multi-TF trend (must agree to be listed)
            mid_trend  = get_trend_direction(series[MID_TF]['c'])
            high_trend = get_trend_direction(series[HIGH_TF]['c'])
            if mid_trend == high_trend and mid_trend in ('bullish','bearish'):
                overall_trend = mid_trend
            else:
                return None  # skip mixed trends entirely (pure bullish/bearish lists)

            # Momentum direction for "Strong/Weak" label
            momentum_trend = 'bullish' if price_change > 0 else 'bearish'

            # Decide label + classification
            label = None
            classification = None
            strength = None

            if indicator_score >= MIN_SCORE_CONFIRMED and abs(price_change) >= PRICE_CHANGE_THRESHOLD:
                label = "(Confirmed)"
                strength = "Strong" if momentum_trend == overall_trend else "Weak"
                classification = f"{overall_trend.capitalize()} {strength}"
                if momentum_score >= 3 and strength == "Strong":
                    classification += " (M)"
            elif indicator_score >= MIN_SCORE_EARLY and momentum_score >= 3 and abs(price_change) >= 2:
                label = "(Early)"
                strength = None  # not used for Early in classification text
                classification = f"{overall_trend.capitalize()} (M)"

            if not label or not classification:
                return None

            # Action hint
            action = derive_action(overall_trend, label, strength or "Weak", momentum_score)

            # list_direction is ALWAYS the multi-TF trend (pure lists)
            list_direction = overall_trend

            indicators_fmt = " - ".join([f"{k}:{'S' if indicators_summary[k] else 'W'}" for k in ['EMA','RSI','MACD','SAR','BOLL','STOCH']])
            msg = (
                f"**{symbol}** {tri}{tag_str} | {price_change:+.2f}% | Price: {last_price:.6g} | "
                f"Score:{indicator_score} | {label} | *{classification}* | {indicators_fmt} | Action: {action}"
            )
            return list_direction, label, indicator_score, abs(price_change), msg
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

def format_ranked_list(entries):
    return "\n\n".join([f"{i+1}. {entry[4]}" for i, entry in enumerate(entries)])

def format_stoch_list(entries):
    return "\n".join([f"{i+1}. {entry[2]}" for i, entry in enumerate(entries)])

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    stoch_tasks = [analyze_stoch(session, s, semaphore) for s in symbols]
    results, stoch_results = await asyncio.gather(asyncio.gather(*tasks), asyncio.gather(*stoch_tasks))

    filtered = [r for r in results if r]
    stoch_filtered = [r for r in stoch_results if r]

    def filter_and_sort(trend, label):
        subset = [r for r in filtered if r[0] == trend and r[1] == label]
        return sorted(subset, key=lambda x: (-x[2], -x[3]))[:10]

    # Pure trend lists (both sides)
    bull_conf = filter_and_sort("bullish", "(Confirmed)")
    bull_early = filter_and_sort("bullish", "(Early)")
    bear_conf = filter_and_sort("bearish", "(Confirmed)")
    bear_early = filter_and_sort("bearish", "(Early)")

    # StochRSI lists with price-change filters
    stoch_bull_all = [r for r in stoch_filtered if r[0] == 'bullish']
    stoch_bull_filtered = [r for r in stoch_bull_all if r[1] >= STOCH_BULL_MIN_RISE]
    stoch_bull = sorted(stoch_bull_filtered, key=lambda x: -x[1])[:10]  # highest positive first

    stoch_bear_all = [r for r in stoch_filtered if r[0] == 'bearish']
    stoch_bear_filtered = [r for r in stoch_bear_all if r[1] <= -STOCH_BEAR_MIN_DROP]
    stoch_bear = sorted(stoch_bear_filtered, key=lambda x: x[1])[:10]    # most negative first

    return bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear

def format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear):
    # Main sections (pure trend)
    msg = "*📊 Binance Futures Trend Scanner*\n\n"
    if bull_conf or bull_early:
        msg += "🚀 *Bullish Trend Coins:*\n"
        if bull_conf:
            msg += "🟢 *Confirmed:*\n" + format_ranked_list(bull_conf) + "\n\n"
        if bull_early:
            msg += "🟡 *Early:*\n" + format_ranked_list(bull_early) + "\n\n"
    if bear_conf or bear_early:
        msg += "🔻 *Bearish Trend Coins:*\n"
        if bear_conf:
            msg += "🔴 *Confirmed:*\n" + format_ranked_list(bear_conf) + "\n\n"
        if bear_early:
            msg += "🟠 *Early:*\n" + format_ranked_list(bear_early) + "\n\n"

    # Visually separated StochRSI section
    if stoch_bull or stoch_bear:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📈 *StochRSI Crossover Signals ({STOCH_TF})*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        if stoch_bull:
            msg += "🟩 *Bullish Cross (≥ +2%):*\n" + format_stoch_list(stoch_bull) + "\n\n"
        if stoch_bear:
            msg += "🟥 *Bearish Cross (≤ -2%):*\n" + format_stoch_list(stoch_bear) + "\n"
    return msg

async def send_telegram(session, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_USER_ID, "text": text, "parse_mode": "Markdown"}
    for _ in range(RETRY_ATTEMPTS):
        try:
            async with session.post(url, data=payload, timeout=HTTP_TIMEOUT) as res:
                if res.status == 200:
                    return True
        except Exception:
            await asyncio.sleep(0.2)
    return False

async def warmup(session):
    # gentle warmup to reduce first-418 probability
    try:
        base = random.choice(BINANCE_FAPI_HOSTS)
        await fetch_json(session, f"{base}/fapi/v1/ping")
        await asyncio.sleep(0.3)
    except Exception:
        pass

async def run_scan():
    async with aiohttp.ClientSession(headers=SESSION_HEADERS) as session:
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        await warmup(session)
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear = await scan_market(session, symbols)
        if any([bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear]):
            msg = format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear)
            await send_telegram(session, msg)
        else:
            await send_telegram(session, "*📊 No coins match filter criteria at the moment.*")
        print("✅ Scan finished.\n")

async def main():
    try:
        await run_scan()
    except Exception as e:
        print("🚨 Error:", e)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
