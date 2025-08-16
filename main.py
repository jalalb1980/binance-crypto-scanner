import asyncio
import aiohttp
import numpy as np
import random
from datetime import datetime
from math import isfinite

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'

MAX_CONCURRENT_REQUESTS = 20
HTTP_TIMEOUT = 15
RETRY_ATTEMPTS = 3
CANDLE_LIMIT = 200

# Prefilter: scan only top-N by 24h quote volume (set to None to scan all)
TOP_N_BY_VOLUME = 120

# Thresholds
MIN_SCORE_EARLY = 3
MIN_SCORE_CONFIRMED = 4
PRICE_CHANGE_THRESHOLD = 2.0
VOLUME_SPIKE_RATIO = 2.0

TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
PRICE_TF = HIGH_TF
TRIANGLE_TF = MID_TF

# StochRSI list
STOCH_TF = HIGH_TF
STOCH_BEAR_MIN_DROP = 2.0
STOCH_BULL_MIN_RISE = 2.0

# Strategy tag toggles
USE_LTR = True
USE_TBB = True
USE_VSQ = True
USE_AVWAP = True

EPS = 1e-12

# --- Hosts / Headers ---
BINANCE_FAPI_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi.binancefuture.com",
    "https://fapi1.binance.com",
]
BASE_HOST = BINANCE_FAPI_HOSTS[0]
SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

# Cache for candles within a single run
CANDLE_CACHE = {}

# === HELPERS ===
def is_futures_usdt(sym):
    return sym.get('quoteAsset') == 'USDT' and sym.get('contractType') == 'PERPETUAL' and sym.get('status') == 'TRADING'

def rewrite_to_base(url: str) -> str:
    for h in BINANCE_FAPI_HOSTS:
        if url.startswith(h):
            return url.replace(h, BASE_HOST)
    return url

async def fetch_json(session, url):
    global BASE_HOST
    last_exc = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            u = rewrite_to_base(url)
            async with session.get(u, timeout=HTTP_TIMEOUT) as res:
                if res.status in (418, 429, 451, 403, 409):
                    others = [h for h in BINANCE_FAPI_HOSTS if h != BASE_HOST]
                    if others:
                        BASE_HOST = random.choice(others)
                    await asyncio.sleep(1.0 + attempt * 0.5)
                    last_exc = f"HTTP {res.status}"
                    continue
                if res.status == 200:
                    return await res.json()
                last_exc = f"HTTP {res.status}"
        except Exception as e:
            last_exc = e
        await asyncio.sleep(0.2 + attempt * 0.3)
    raise RuntimeError(f"GET failed: {url} -> {last_exc}")

async def choose_base_host(session):
    global BASE_HOST
    for host in BINANCE_FAPI_HOSTS:
        try:
            async with session.get(f"{host}/fapi/v1/ping", timeout=HTTP_TIMEOUT) as r:
                if r.status == 200:
                    BASE_HOST = host
                    return
        except:
            continue
    BASE_HOST = BINANCE_FAPI_HOSTS[0]

async def fetch_symbols(session):
    data = await fetch_json(session, f"{BASE_HOST}/fapi/v1/exchangeInfo")
    return [s['symbol'] for s in data['symbols'] if is_futures_usdt(s)]

async def prefilter_top_by_volume(session, symbols, top_n=TOP_N_BY_VOLUME):
    if not top_n:
        return symbols
    data = await fetch_json(session, f"{BASE_HOST}/fapi/v1/ticker/24hr")
    rows = [d for d in data if d.get('symbol') in symbols]
    def qv(d):
        try:
            return float(d.get('quoteVolume') or 0.0)
        except:
            return 0.0
    rows.sort(key=qv, reverse=True)
    picks = [d['symbol'] for d in rows[:top_n]]
    return picks if picks else symbols

async def fetch_candles(session, symbol, interval, limit=CANDLE_LIMIT):
    key = (symbol, interval, limit)
    if key in CANDLE_CACHE:
        return CANDLE_CACHE[key]
    url = f"{BASE_HOST}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = await fetch_json(session, url)
    CANDLE_CACHE[key] = data
    return data

def to_ohlcv(candles):
    o = np.array([float(x[1]) for x in candles], dtype=float)
    h = np.array([float(x[2]) for x in candles], dtype=float)
    l = np.array([float(x[3]) for x in candles], dtype=float)
    c = np.array([float(x[4]) for x in candles], dtype=float)
    v = np.array([float(x[5]) for x in candles], dtype=float)
    return o, h, l, c, v

# === INDICATORS ===
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
    avg_gain[:period] = np.nan;       avg_loss[:period] = np.nan
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
    if len(c) < slow + signal + 5: return np.nan
    ema_fast = ema(c, fast); ema_slow = ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return float((macd_line - signal_line)[-1])

def bollinger_bands_signal(closes, period=20, dev=2.0):
    c = np.asarray(closes, dtype=float)
    if len(c) < period: return False
    ma = c[-period:].mean(); sd = c[-period:].std(ddof=0)
    upper = ma + dev*sd; lower = ma - dev*sd
    return bool(c[-1] > upper or c[-1] < lower)

def psar_signal(highs, lows, closes, af_step=0.02, af_max=0.2):
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if n < 5: return c[-1] > c[-2]
    uptrend = c[1] > c[0]
    ep = h[0] if uptrend else l[0]; af = af_step; psar = c[0]
    for i in range(1, n):
        prev_psar = psar
        if uptrend:
            psar = prev_psar + af * (ep - prev_psar)
            psar = min(psar, l[i-1], l[i-2] if i >= 2 else l[i-1])
            if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
            if l[i] < psar: uptrend=False; psar=ep; ep=l[i]; af=af_step
        else:
            psar = prev_psar + af * (ep - prev_psar)
            psar = max(psar, h[i-1], h[i-2] if i >= 2 else h[i-1])
            if l[i] < ep: ep=l[i]; af=min(af + af_step, af_max)
            if h[i] > psar: uptrend=True; psar=ep; ep=h[i]; af=af_step
    return bool(c[-1] > psar)

def stoch_rsi_signal(closes, period=14):
    r = rsi_wilder(closes, period)
    valid = r[~np.isnan(r)]
    if len(valid) < period + 3: return None
    rsi_window = valid[-period:]
    rmin = np.min(rsi_window); rmax = np.max(rsi_window)
    denom = max(rmax - rmin, EPS)
    k = (valid[-1] - rmin) / denom * 100.0
    k_series = []
    for back in range(3, 0, -1):
        win = valid[-(period + (3 - back)) : -(3 - back) if (3 - back) != 0 else None]
        if len(win) < period: k_series.append(k); continue
        rmin_b = np.min(win[-period:]); rmax_b = np.max(win[-period:])
        denom_b = max(rmax_b - rmin_b, EPS)
        k_series.append((win[-1] - rmin_b) / denom_b * 100.0)
    d = float(np.mean(k_series)); diff = abs(k - d)
    signal = 'bullish' if (k > d and k < 40) else ('bearish' if (k < d and k > 60) else None)
    level = 'Hot' if diff <= 3 else 'Good' if diff <= 15 else 'Normal'
    return signal, diff, level, k, d

def indicator_bundle(highs, lows, closes):
    ema_fast_up = ema(closes, 9); ema_slow_up = ema(closes, 21)
    ema_ok = bool(ema_fast_up.size and ema_slow_up.size and ema_fast_up[-1] > ema_slow_up[-1])
    rsi_arr = rsi_wilder(closes, 14)
    rsi_ok = bool(rsi_arr.size and isfinite(rsi_arr[-1]) and rsi_arr[-1] > 50)
    mh = macd_histogram(closes); macd_ok = bool(isfinite(mh) and mh > 0)
    sar_ok = psar_signal(highs, lows, closes)
    boll_ok = bollinger_bands_signal(closes)
    stoch = stoch_rsi_signal(closes); stoch_ok = (stoch[0] == 'bullish') if stoch else False
    return {'EMA': ema_ok,'RSI': rsi_ok,'MACD': macd_ok,'SAR': sar_ok,'BOLL': boll_ok,'STOCH': stoch_ok}

# === THESE TWO MUST EXIST BEFORE analyze_symbol ===
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
    if len(e9) < 1 or len(e21) < 1 or len(rsi_arr) < 1: return 0, False
    vol_spike = volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * VOLUME_SPIKE_RATIO if len(volumes) > 20 else False
    score = 0
    if e9[-1] > e21[-1]: score += 1
    if rsi_arr[-1] > 50: score += 1
    if isfinite(macd_h) and macd_h > 0: score += 1
    if vol_spike: score += 1
    return score, vol_spike

# === TAGS (optional) ===
def detect_triangle_tag(closes, highs, lows, lookback=30, tight=0.02):
    if len(closes) < lookback: return '(Tx)'
    hb = float(np.max(highs[-lookback:])); lb = float(np.min(lows[-lookback:]))
    width = hb - lb
    if width <= 0: return '(Tx)'
    if width / max(closes[-1], EPS) < tight:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return '(Tx)'

def tag_liquidity_trap_reversal(closes, highs, lows, volumes):
    if not USE_LTR or len(closes) < 6: return None
    range_hl = highs[-1] - lows[-1]
    if range_hl <= 0: return None
    vol_spike = volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * 2.0 if len(volumes) > 20 else False
    tail = closes[-1] - lows[-1]
    if (tail / max(range_hl, EPS) > 0.6) and vol_spike: return 'LTR▲'
    upper_wick = highs[-1] - closes[-1]
    if (upper_wick / max(range_hl, EPS) > 0.6) and vol_spike: return 'LTR▼'
    return None

def tag_time_based_breakout(closes, opens, volumes, timestamps=None, compress_len=20, width_pct=0.01):
    if not USE_TBB or len(closes) < compress_len + 5: return None
    window = closes[-compress_len:]
    w = (np.max(window) - np.min(window)) / max(window[-1], EPS)
    if w < width_pct:
        prev_max = np.max(window[:-1]); prev_min = np.min(window[:-1])
        if closes[-1] > prev_max and volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * 1.5: return 'TBB▲'
        if closes[-1] < prev_min and volumes[-1] > (np.mean(volumes[-20:-1]) + EPS) * 1.5: return 'TBB▼'
    return None

def tag_volatility_squeeze(closes, period=20, dev=2.0, pct_thresh=0.08):
    if not USE_VSQ or len(closes) < period + 5: return None
    c = np.asarray(closes, dtype=float)
    ma = c[-period:].mean(); sd = c[-period:].std(ddof=0)
    width = (2 * dev * sd) / max(ma, EPS)
    if width < pct_thresh:
        if c[-1] > ma + dev*sd: return 'VSQ▲'
        if c[-1] < ma - dev*sd: return 'VSQ▼'
    return None

def tag_avwap_reclaim(closes, highs, lows, anchor_lookback=60):
    if not USE_AVWAP or len(closes) < anchor_lookback + 5: return None
    swing_low_idx = np.argmin(lows[-anchor_lookback:])
    swing_high_idx = np.argmax(highs[-anchor_lookback:])
    def avwap_from(idx):
        start = len(closes) - anchor_lookback + idx
        c = np.array(closes[start:], dtype=float); v = np.ones_like(c)
        return float(np.sum(c * v) / np.sum(v))
    avwap_low = avwap_from(swing_low_idx); avwap_high = avwap_from(swing_high_idx)
    if closes[-1] > avwap_low and closes[-2] <= avwap_low: return 'AVWAP▲'
    if closes[-1] < avwap_high and closes[-2] >= avwap_high: return 'AVWAP▼'
    return None

# === ACTION HINT ===
def derive_action(overall_trend, label, strength, momentum_score):
    if overall_trend == 'bullish':
        if label == "(Confirmed)": return "LONG (A)" if strength == "Strong" else "LONG (B)"
        return "LONG (B)"
    else:
        if label == "(Confirmed)": return "SHORT (A)" if strength == "Strong" else "SHORT (B)"
        return "SHORT (B)"

# === ANALYSIS ===
async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = list(set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF, STOCH_TF]))
            tasks = [fetch_candles(session, symbol, tf) for tf in tfs]
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            candles = {}
            for tf, data in zip(tfs, fetched):
                if isinstance(data, Exception): return None
                candles[tf] = data

            series = {}
            for tf in tfs:
                o,h,l,c,v = to_ohlcv(candles[tf])
                series[tf] = {'o':o,'h':h,'l':l,'c':c,'v':v}

            last_price = float(series[PRICE_TF]['c'][-1])
            price_change = (series[PRICE_TF]['c'][-1] - series[PRICE_TF]['c'][-2]) / max(series[PRICE_TF]['c'][-2], EPS) * 100.0

            indicators_summary = {}
            for tf in INDICATOR_TFS:
                ind = indicator_bundle(series[tf]['h'], series[tf]['l'], series[tf]['c'])
                for k, v in ind.items():
                    indicators_summary[k] = indicators_summary.get(k, 0) + (1 if v else 0)
            indicator_score = sum(1 for v in indicators_summary.values() if v > 0)

            momentum_score, _ = detect_momentum(series[MOMENTUM_TF]['c'], series[MOMENTUM_TF]['v'])

            tri = detect_triangle_tag(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['h'], series[TRIANGLE_TF]['l'])
            tags = []
            for t in (
                tag_liquidity_trap_reversal(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['h'], series[TRIANGLE_TF]['l'], series[TRIANGLE_TF]['v']),
                tag_time_based_breakout(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['o'], series[TRIANGLE_TF]['v']),
                tag_volatility_squeeze(series[TRIANGLE_TF]['c']),
                tag_avwap_reclaim(series[TRIANGLE_TF]['c'], series[TRIANGLE_TF]['h'], series[TRIANGLE_TF]['l'])
            ):
                if t: tags.append(t)
            tag_str = (" [" + ",".join(tags) + "]") if tags else ""

            mid_trend  = get_trend_direction(series[MID_TF]['c'])
            high_trend = get_trend_direction(series[HIGH_TF]['c'])
            if mid_trend == high_trend and mid_trend in ('bullish','bearish'):
                overall_trend = mid_trend
            else:
                return None

            momentum_trend = 'bullish' if price_change > 0 else 'bearish'

            label = None; classification = None; strength = None
            if indicator_score >= MIN_SCORE_CONFIRMED and abs(price_change) >= PRICE_CHANGE_THRESHOLD:
                label = "(Confirmed)"
                strength = "Strong" if momentum_trend == overall_trend else "Weak"
                classification = f"{overall_trend.capitalize()} {strength}"
                if momentum_score >= 3 and strength == "Strong":
                    classification += " (M)"
            elif indicator_score >= MIN_SCORE_EARLY and momentum_score >= 3 and abs(price_change) >= 2:
                label = "(Early)"
                strength = "Weak"
                classification = f"{overall_trend.capitalize()} (M)"
            if not label: return None

            action = derive_action(overall_trend, label, strength, momentum_score)

            indicators_fmt = " - ".join([f"{k}:{'S' if indicators_summary[k] else 'W'}" for k in ['EMA','RSI','MACD','SAR','BOLL','STOCH']])
            msg = (
                f"**{symbol}** {tri}{tag_str} | {price_change:+.2f}% | Price: {last_price:.6g} | "
                f"Score:{indicator_score} | {label} | *{classification}* | {indicators_fmt} | Action: {action}"
            )

            stoch = stoch_rsi_signal(series[STOCH_TF]['c'])
            stoch_entry = None
            if stoch:
                sgn, diff, level, k, d = stoch
                pc = (series[STOCH_TF]['c'][-1] - series[STOCH_TF]['c'][-2]) / max(series[STOCH_TF]['c'][-2], EPS) * 100.0
                lab = f"**{symbol}** | {pc:+.2f}% | {sgn.upper()} | K:{k:.1f} D:{d:.1f} | Δ:{diff:.2f} | 🔥 {level}"
                stoch_entry = (sgn, pc, lab)

            return (overall_trend, label, indicator_score, abs(price_change), msg), stoch_entry
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
    results = await asyncio.gather(*tasks)

    trend_items, stoch_items = [], []
    for r in results:
        if not r: continue
        trend_part, stoch_part = r
        if trend_part: trend_items.append(trend_part)
        if stoch_part: stoch_items.append(stoch_part)

    def filter_and_sort(trend, label):
        subset = [r for r in trend_items if r[0] == trend and r[1] == label]
        return sorted(subset, key=lambda x: (-x[2], -x[3]))[:10]

    bull_conf = filter_and_sort("bullish", "(Confirmed)")
    bull_early = filter_and_sort("bullish", "(Early)")
    bear_conf = filter_and_sort("bearish", "(Confirmed)")
    bear_early = filter_and_sort("bearish", "(Early)")

    stoch_bull = sorted([r for r in stoch_items if r and r[0]=='bullish' and r[1]>=STOCH_BULL_MIN_RISE], key=lambda x: -x[1])[:10]
    stoch_bear = sorted([r for r in stoch_items if r and r[0]=='bearish' and r[1]<=-STOCH_BEAR_MIN_DROP], key=lambda x: x[1])[:10]

    return bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear

def format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear):
    msg = "*📊 Binance Futures Trend Scanner*\n\n"
    if bull_conf or bull_early:
        msg += "🚀 *Bullish Trend Coins:*\n"
        if bull_conf: msg += "🟢 *Confirmed:*\n" + format_ranked_list(bull_conf) + "\n\n"
        if bull_early: msg += "🟡 *Early:*\n" + format_ranked_list(bull_early) + "\n\n"
    if bear_conf or bear_early:
        msg += "🔻 *Bearish Trend Coins:*\n"
        if bear_conf: msg += "🔴 *Confirmed:*\n" + format_ranked_list(bear_conf) + "\n\n"
        if bear_early: msg += "🟠 *Early:*\n" + format_ranked_list(bear_early) + "\n\n"
    if stoch_bull or stoch_bear:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📈 *StochRSI Crossover Signals ({STOCH_TF})*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        if stoch_bull: msg += "🟩 *Bullish Cross (≥ +2%):* \n" + format_stoch_list(stoch_bull) + "\n\n"
        if stoch_bear: msg += "🟥 *Bearish Cross (≤ -2%):* \n" + format_stoch_list(stoch_bear) + "\n"
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

async def run_scan():
    async with aiohttp.ClientSession(headers=SESSION_HEADERS) as session:
        await choose_base_host(session)
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning on {BASE_HOST}...")
        symbols = await fetch_symbols(session)
        symbols = await prefilter_top_by_volume(session, symbols, TOP_N_BY_VOLUME)
        CANDLE_CACHE.clear()
        bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear = await scan_market(session, symbols)
        if any([bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear]):
            await send_telegram(session, format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear))
        else:
            await send_telegram(session, "*📊 No coins match filter criteria at the moment.*")
        print("✅ Scan finished.\n")

async def main():
    try:
        # quick sanity check
        if not callable(detect_momentum) or not callable(get_trend_direction):
            print("⚠️ Sanity check failed: missing detect_momentum/get_trend_direction.")
        await run_scan()
    except Exception as e:
        print("🚨 Error:", e)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
