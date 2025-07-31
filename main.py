import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
MAX_CONCURRENT_REQUESTS = 50
CANDLE_LIMIT = 50
PRICE_CHANGE_THRESHOLD = 2.0
VOLUME_SPIKE_RATIO = 2.0
MIN_SCORE_EARLY = 3
MIN_SCORE_CONFIRMED = 4

TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
PRICE_TF = "1h"
TRIANGLE_TF = MID_TF

# === UTILS ===
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
def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0

def calc_ema(closes, span):
    if len(closes) < span:
        return closes[-1]
    weights = np.exp(np.linspace(-1., 0., span))
    weights /= weights.sum()
    a = np.convolve(closes, weights, mode='full')[:len(closes)]
    return a[-1]

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    delta = np.diff(closes)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = safe_mean(gain[-period:])
    avg_loss = safe_mean(loss[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(closes):
    if len(closes) < 26:
        return 0
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    return ema12 - ema26

def psar(closes):
    return closes[-1] > closes[-2] if len(closes) >= 2 else False

def bollinger_band(closes, period=20):
    if len(closes) < period:
        return False
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return closes[-1] > ma + std or closes[-1] < ma - std

def stochrsi_logic(closes, period=14):
    if len(closes) < period * 2:
        return False
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes) - period)]
    if len(rsi_vals) < 2:
        return False
    last = rsi_vals[-1]
    prev = rsi_vals[-2]
    below_50 = all(r < 50 for r in rsi_vals[-3:])
    above_70 = all(r > 70 for r in rsi_vals[-3:])
    return (below_50 and last > prev) or (above_70 and last < prev)

def detect_momentum(closes, volumes):
    if len(closes) < 10 or len(volumes) < 2:
        return False
    ema_fast = calc_ema(closes[-10:], 9)
    ema_slow = calc_ema(closes[-10:], 21)
    rsi_val = rsi(closes)
    macd_hist = macd(closes)
    vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO
    return ema_fast > ema_slow and rsi_val > 50 and macd_hist > 0 and vol_spike

def detect_triangle(candles):
    try:
        highs = [float(c[2]) for c in candles[-10:]]
        lows = [float(c[3]) for c in candles[-10:]]
        closes = [float(c[4]) for c in candles[-10:]]
        upper = max(highs)
        lower = min(lows)
        width = upper - lower
        if width / closes[-1] < 0.02:
            return '▲' if closes[-1] > closes[-2] else '▼'
    except:
        return "(Tx)"
    return "(Tx)"

# === MAIN ANALYSIS ===
async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]

            indicators_summary = {}
            for tf in INDICATOR_TFS:
                data = closes[tf]
                indicators = {
                    'EMA': calc_ema(data, 9) > calc_ema(data, 21),
                    'RSI': rsi(data) > 50,
                    'MACD': macd(data) > 0,
                    'SAR': psar(data),
                    'BOLL': bollinger_band(data),
                    'STOCHRSI': stochrsi_logic(data)
                }
                for k, v in indicators.items():
                    indicators_summary.setdefault(k, []).append(v)

            final_indicators = {k: sum(v) for k, v in indicators_summary.items()}
            score = sum(v > 0 for v in final_indicators.values())
            momentum = detect_momentum(closes[MOMENTUM_TF], volumes)
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100
            triangle = detect_triangle(candles[TRIANGLE_TF])
            vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO

            label = None
            if score >= MIN_SCORE_CONFIRMED and abs(price_change) >= PRICE_CHANGE_THRESHOLD:
                label = "(Confirmed)"
            elif score >= MIN_SCORE_EARLY and momentum and abs(price_change) >= 2:
                label = "(Early)"
            if not label:
                return None

            trend = "bullish" if price_change > 0 else "bearish"
            indicators_fmt = " - ".join([f"{k}:{'S' if final_indicators[k] else 'W'}" for k in final_indicators])
            msg = f"**{symbol}** {triangle}{' (M)' if momentum else ''}{' Vol↑' if vol_spike else ''} | {price_change:+.2f}% | Score:{score} | {label} | {indicators_fmt}"
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
    return msg or "📭 No valid coins found this round."

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    results = await asyncio.gather(*tasks)
    filtered = [r for r in results if r]

    def filter_and_sort(trend, label):
        return sorted([r for r in filtered if r[0] == trend and r[1] == label], key=lambda x: (-x[2], -x[3]))[:10]

    return (
        filter_and_sort("bullish", "(Early)"),
        filter_and_sort("bullish", "(Confirmed)"),
        filter_and_sort("bearish", "(Early)"),
        filter_and_sort("bearish", "(Confirmed)")
    )

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf = await scan_market(session, symbols)
        msg = format_report(bull_early, bull_conf, bear_early, bear_conf)
        await send_telegram(session, msg)
        print("✅ Scan finished.\n")

async def main():
    try:
        await run_scan()
    except Exception as e:
        print(f"🚨 Error: {e}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
