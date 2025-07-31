import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
SLEEP_INTERVAL = 1800
MAX_CONCURRENT_REQUESTS = 50
MIN_SCORE_EARLY = 3
MIN_SCORE_CONFIRMED = 4
VOLUME_SPIKE_RATIO = 2.0
CANDLE_LIMIT = 50
PRICE_SORT_THRESHOLD = 2.0

TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
PRICE_TF = "1h"
TRIANGLE_TF = MID_TF

# === HELPER FUNCTIONS ===
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
    a = np.convolve(closes, weights, mode='full')[:len(closes)]
    return a[-1]

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

def macd_histogram(closes):
    ema12 = calc_ema(closes[-26:], 12)
    ema26 = calc_ema(closes[-26:], 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(closes[-9:], 9)
    return macd_line - signal_line

def psar(closes):
    return closes[-1] > closes[-2]  # Simplified trend check

def bollinger_band(closes, period=20):
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return closes[-1] > ma + std or closes[-1] < ma - std

def stochrsi_logic(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes)-period)]
    if len(rsi_vals) < 2:
        return None
    last = rsi_vals[-1]
    prev = rsi_vals[-2]
    bullish = last > prev and last < 50 and prev < 50
    bearish = last < prev and last > 50 and prev > 50
    return 'bullish' if bullish else 'bearish' if bearish else None

def detect_triangle(candles):
    highs = [float(c[2]) for c in candles[-10:]]
    lows = [float(c[3]) for c in candles[-10:]]
    closes = [float(c[4]) for c in candles[-10:]]
    width = max(highs) - min(lows)
    if width / closes[-1] < 0.02:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return ''

def detect_momentum(closes, volumes, direction):
    score = 0
    ema_fast = calc_ema(closes[-10:], 9)
    ema_slow = calc_ema(closes[-10:], 21)
    rsi_val = rsi(closes)
    macd_val = macd_histogram(closes)
    vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO

    if direction == 'bullish':
        if ema_fast > ema_slow: score += 1
        if rsi_val > 50: score += 1
        if macd_val > 0: score += 1
        if vol_spike: score += 1
    else:
        if ema_fast < ema_slow: score += 1
        if rsi_val < 50: score += 1
        if macd_val < 0: score += 1
        if vol_spike: score += 1

    return score

# === ANALYSIS ===
async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MOMENTUM_TF]]

            combined = []
            for tf in INDICATOR_TFS:
                ind = closes[tf]
                indicators = {
                    'EMA': calc_ema(ind, 9) > calc_ema(ind, 21),
                    'RSI': rsi(ind) > 50,
                    'MACD': macd_histogram(ind) > 0,
                    'SAR': psar(ind),
                    'BOLL': bollinger_band(ind),
                    'STOCH': stochrsi_logic(ind) == 'bullish'
                }
                combined.append(indicators)

            summary = {k: sum(i[k] for i in combined) for k in combined[0]}
            score = sum(v > 0 for v in summary.values())

            # Determine price trend
            price_now = closes[PRICE_TF][-1]
            price_prev = closes[PRICE_TF][-2]
            price_change = ((price_now - price_prev) / price_prev) * 100
            trend = 'bullish' if price_change > 0 else 'bearish'

            triangle = detect_triangle(candles[TRIANGLE_TF])
            momentum_score = detect_momentum(closes[MOMENTUM_TF], volumes, trend)
            avg3 = np.mean(closes[PRICE_TF][-3:])

            if score < MIN_SCORE_EARLY or momentum_score < 2:
                return None

            label = "(Confirmed)" if score >= MIN_SCORE_CONFIRMED else "(Early)"
            indicators_fmt = " - ".join([f"{k}:{'S' if summary[k] else 'W'}" for k in summary])
            msg = f"**{symbol}** {triangle} (M) | Δ{price_change:+.2f}% | Score:{score} | {label} | MA3:{avg3:.2f} | {indicators_fmt}"
            return trend, label, score, abs(price_change), msg

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

def format_ranked_list(entries):
    return "\n\n".join([f"{i+1}. {entry[4]}" for i, entry in enumerate(entries)])

def format_report(bull_early, bull_conf, bear_early, bear_conf):
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
            msg += "🟠 *Early:*\n" + format_ranked_list(bear_early)
    return msg

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    results = await asyncio.gather(*tasks)
    filtered = [r for r in results if r]

    def filter_and_sort(trend, label):
        return sorted(
            [r for r in filtered if r[0] == trend and r[1] == label],
            key=lambda x: (-x[2], -x[3])
        )[:10]

    bull_conf = filter_and_sort("bullish", "(Confirmed)")
    bull_early = filter_and_sort("bullish", "(Early)")
    bear_conf = filter_and_sort("bearish", "(Confirmed)")
    bear_early = filter_and_sort("bearish", "(Early)")

    return bull_early, bull_conf, bear_early, bear_conf

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf = await scan_market(session, symbols)
        if bull_early or bull_conf or bear_early or bear_conf:
            msg = format_report(bull_early, bull_conf, bear_early, bear_conf)
            await send_telegram(session, msg)
        print("✅ Scan complete.")

async def main():
    try:
        await run_scan()
    except Exception as e:
        print("🚨 Error:", e)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
