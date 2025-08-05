import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
MAX_CONCURRENT_REQUESTS = 50
CANDLE_LIMIT = 50

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

def is_futures_usdt(symbol):
    return symbol.get('quoteAsset') == 'USDT' and symbol.get('contractType') == 'PERPETUAL'

async def fetch_symbols(session):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    async with session.get(url) as res:
        data = await res.json()
        return [s['symbol'] for s in data['symbols'] if is_futures_usdt(s)]

async def fetch_candles(session, symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={CANDLE_LIMIT}"
    async with session.get(url) as res:
        return await res.json()

async def send_telegram(session, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url, data={"chat_id": TELEGRAM_USER_ID, "text": text, "parse_mode": "Markdown"})

# === INDICATORS ===
def calc_ema(closes, span):
    weights = np.exp(np.linspace(-1., 0., span))
    weights /= weights.sum()
    return np.convolve(closes, weights, mode='full')[:len(closes)][-1]

def rsi(closes, period=14):
    delta = np.diff(closes)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_hist(closes):
    ema12 = calc_ema(closes[-26:], 12)
    ema26 = calc_ema(closes[-26:], 26)
    macd = ema12 - ema26
    signal = calc_ema(closes[-9:], 9)
    return macd - signal

def psar(closes):
    return closes[-1] > closes[-2]

def bollinger_band(closes, period=20):
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return closes[-1] > ma + std or closes[-1] < ma - std

def detect_triangle(candles):
    highs = [float(c[2]) for c in candles[-10:]]
    lows = [float(c[3]) for c in candles[-10:]]
    closes = [float(c[4]) for c in candles[-10:]]
    width = max(highs) - min(lows)
    if width / closes[-1] < 0.02:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return '(Tx)'

def stochrsi_crossover(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes) - period)]
    if len(rsi_vals) < 3:
        return None
    k = rsi_vals[-1]
    d = np.mean(rsi_vals[-3:])
    if k > d and k < 20 and d < 20:
        return 'bullish'
    elif k < d and k > 80 and d > 80:
        return 'bearish'
    return None

def detect_momentum(closes, volumes):
    ema_fast = calc_ema(closes[-10:], 9)
    ema_slow = calc_ema(closes[-10:], 21)
    rsi_val = rsi(closes)
    macd = macd_hist(closes)
    vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO
    score = 0
    if ema_fast > ema_slow: score += 1
    if rsi_val > 50: score += 1
    if macd > 0: score += 1
    if vol_spike: score += 1
    return score, vol_spike

def get_trend_direction(closes):
    ema_fast = calc_ema(closes, 9)
    ema_slow = calc_ema(closes, 21)
    if ema_fast > ema_slow:
        return 'bullish'
    elif ema_fast < ema_slow:
        return 'bearish'
    return 'neutral'

async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]
            last_price = closes[PRICE_TF][-1]

            indicators_summary = {}
            for tf in INDICATOR_TFS:
                ind = closes[tf]
                indicators = {
                    'EMA': calc_ema(ind, 9) > calc_ema(ind, 21),
                    'RSI': rsi(ind) > 50,
                    'MACD': macd_hist(ind) > 0,
                    'SAR': psar(ind),
                    'BOLL': bollinger_band(ind),
                }
                for k, v in indicators.items():
                    indicators_summary[k] = indicators_summary.get(k, 0) + int(v)

            indicator_score = sum(1 for v in indicators_summary.values() if v > 0)
            momentum_score, vol_spike = detect_momentum(closes[MOMENTUM_TF], volumes)
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100
            triangle = detect_triangle(candles[TRIANGLE_TF])

            if indicator_score < MIN_SCORE_EARLY:
                return None

            mid_trend = get_trend_direction(closes[MID_TF])
            high_trend = get_trend_direction(closes[HIGH_TF])
            overall_trend = 'mixed'
            if mid_trend == high_trend and mid_trend in ['bullish', 'bearish']:
                overall_trend = mid_trend

            momentum_trend = 'bullish' if price_change > 0 else 'bearish'

            label = None
            classification = None

            if indicator_score >= MIN_SCORE_CONFIRMED and abs(price_change) >= PRICE_CHANGE_THRESHOLD:
                label = "(Confirmed)"
                classification = f"{momentum_trend.capitalize()} {'Strong' if overall_trend == momentum_trend else 'Weak'}"
                if momentum_score >= 3:
                    classification += " (M)"
            elif indicator_score >= MIN_SCORE_EARLY and momentum_score >= 3 and abs(price_change) >= 2:
                label = "(Early)"
                classification = f"{momentum_trend.capitalize()} (M)"

            if not label or not classification:
                return None

            indicators_fmt = " - ".join([f"{k}:{'S' if indicators_summary[k] else 'W'}" for k in indicators_summary])
            msg = f"**{symbol}** {triangle} | {price_change:+.2f}% | Price: {last_price:.2f} | Score:{indicator_score} | {label} | *{classification}* | {indicators_fmt}"
            return momentum_trend, label, indicator_score, abs(price_change), msg
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

async def scan_stochrsi(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = []

    async def task(symbol):
        async with semaphore:
            try:
                candles = await fetch_candles(session, symbol, "4h")
                closes = [float(c[4]) for c in candles]
                signal = stochrsi_crossover(closes)
                if signal:
                    return f"**{symbol}** | StochRSI: *{signal.capitalize()}*"
            except:
                return None

    tasks = [task(s) for s in symbols]
    raw = await asyncio.gather(*tasks)
    return sorted([r for r in raw if r])[:10]

def format_ranked_list(entries):
    return "\n\n".join([f"{i+1}. {entry[4]}" for i, entry in enumerate(entries)])

def format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_signals):
    msg = "*📊 Binance Futures Trend Scanner*\n\n"
    if bull_conf or bull_early:
        msg += "🚀 *Bullish Signals:*\n"
        if bull_conf:
            msg += "🟢 *Confirmed:*\n" + format_ranked_list(bull_conf) + "\n\n"
        if bull_early:
            msg += "🟡 *Early:*\n" + format_ranked_list(bull_early) + "\n\n"
    if bear_conf or bear_early:
        msg += "🔻 *Bearish Signals:*\n"
        if bear_conf:
            msg += "🔴 *Confirmed:*\n" + format_ranked_list(bear_conf) + "\n\n"
        if bear_early:
            msg += "🟠 *Early:*\n" + format_ranked_list(bear_early) + "\n\n"
    if stoch_signals:
        msg += "📈 *StochRSI Crossover Signals:*\n" + "\n".join(stoch_signals)
    return msg

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    results = await asyncio.gather(*tasks)
    filtered = [r for r in results if r]

    def filter_and_sort(trend, label):
        return sorted([r for r in filtered if r[0] == trend and r[1] == label], key=lambda x: (-x[2], -x[3]))[:10]

    bull_conf = filter_and_sort("bullish", "(Confirmed)")
    bull_early = filter_and_sort("bullish", "(Early)")
    bear_conf = filter_and_sort("bearish", "(Confirmed)")
    bear_early = filter_and_sort("bearish", "(Early)")

    return bull_early, bull_conf, bear_early, bear_conf

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf = await scan_market(session, symbols)
        stoch_signals = await scan_stochrsi(session, symbols)
        msg = format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_signals)
        await send_telegram(session, msg if msg else "*📊 No signals detected.*")
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
