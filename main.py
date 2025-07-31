import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
CANDLE_LIMIT = 50
MAX_CONCURRENT_REQUESTS = 50

TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
PRICE_TF = "1h"  # updated as per your request
TRIANGLE_TF = MID_TF

MIN_SCORE_EARLY = 3
MIN_SCORE_CONFIRMED = 4
VOLUME_SPIKE_RATIO = 2.0

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
    return 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

def macd(closes):
    ema12 = calc_ema(closes[-26:], 12)
    ema26 = calc_ema(closes[-26:], 26)
    return ema12 - ema26

def psar(closes):
    return closes[-1] > closes[-2]

def bollinger_band(closes, period=20):
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return closes[-1] > ma + std or closes[-1] < ma - std

def stochrsi(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes)-period)]
    if len(rsi_vals) < 2:
        return False
    stoch = rsi_vals[-1]
    prev_stoch = rsi_vals[-2]
    avg = np.mean(rsi_vals[-3:])
    if stoch < 50 and prev_stoch < stoch:
        return True  # bullish
    if stoch > 50 and prev_stoch > stoch:
        return True  # bearish
    return False

def detect_triangle(candles):
    highs = [float(c[2]) for c in candles[-10:]]
    lows = [float(c[3]) for c in candles[-10:]]
    closes = [float(c[4]) for c in candles[-10:]]
    if not closes:
        return "(Tx)"
    width = max(highs) - min(lows)
    if width / closes[-1] < 0.02:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return '(Tx)'

def detect_momentum(closes, volumes):
    ema_fast = calc_ema(closes[-10:], 9)
    ema_slow = calc_ema(closes[-10:], 21)
    rsi_val = rsi(closes)
    macd_val = macd(closes)
    vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO
    return ema_fast > ema_slow and rsi_val > 50 and macd_val > 0 and vol_spike

def is_clean_trend(closes, bullish=True):
    if len(closes) < 4:
        return False
    ma3 = np.mean(closes[-3:])
    return (closes[-1] > ma3) if bullish else (closes[-1] < ma3)

async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]

            summary = {}
            for key in ['EMA', 'RSI', 'MACD', 'SAR', 'BOLL', 'STOCH']:
                summary[key] = 0
                for tf in INDICATOR_TFS:
                    data = closes[tf]
                    if key == 'EMA':
                        summary[key] += calc_ema(data, 9) > calc_ema(data, 21)
                    elif key == 'RSI':
                        summary[key] += rsi(data) > 50
                    elif key == 'MACD':
                        summary[key] += macd(data) > 0
                    elif key == 'SAR':
                        summary[key] += psar(data)
                    elif key == 'BOLL':
                        summary[key] += bollinger_band(data)
                    elif key == 'STOCH':
                        summary[key] += stochrsi(data)

            score = sum(1 for v in summary.values() if v > 0)
            momentum = detect_momentum(closes[MOMENTUM_TF], volumes)
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100
            triangle = detect_triangle(candles[TRIANGLE_TF])
            vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO
            is_bull = price_change > 0
            clean_trend = is_clean_trend(closes[MID_TF], bullish=is_bull)

            if score < MIN_SCORE_EARLY or not clean_trend or abs(price_change) < 2 or not vol_spike:
                return None

            label = "(Confirmed)" if score >= MIN_SCORE_CONFIRMED else "(Early)"
            trend = "bullish" if is_bull else "bearish"
            indicators_fmt = " - ".join([f"{k}:{'S' if summary[k] else 'W'}" for k in summary])
            msg = f"**{symbol}** {triangle}{' (M)' if momentum else ''}{' Vol↑' if vol_spike else ''} | {price_change:+.2f}% | Score:{score} | {label} | {indicators_fmt}"
            return trend, label, score, abs(price_change), msg
        except Exception as e:
            print(f"❌ Error {symbol}: {e}")
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

    def filter_sort(trend, label):
        return sorted([r for r in filtered if r[0] == trend and r[1] == label], key=lambda x: (-x[2], -x[3]))[:10]

    return filter_sort("bullish", "(Early)"), filter_sort("bullish", "(Confirmed)"), filter_sort("bearish", "(Early)"), filter_sort("bearish", "(Confirmed)")

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"🔍 Scanning at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf = await scan_market(session, symbols)
        if bull_early or bull_conf or bear_early or bear_conf:
            msg = format_report(bull_early, bull_conf, bear_early, bear_conf)
            await send_telegram(session, msg)
        print("✅ Scan done.")

async def main():
    try:
        await run_scan()
    except Exception as e:
        print("🚨 Error:", e)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
