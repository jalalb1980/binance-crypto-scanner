import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
TELEGRAM_USER_ID = 'YOUR_USER_ID'
MAX_CONCURRENT_REQUESTS = 50
CANDLE_LIMIT = 50
VOLUME_SPIKE_RATIO = 2.0
PRICE_CHANGE_THRESHOLD = 2.0
MIN_SCORE = 4  # min indicator score
TIMEFRAMES = ["30m", "1h", "4h"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = "30m"
INDICATOR_TFS = ["1h", "4h"]
PRICE_TF = "1h"
TRIANGLE_TF = "1h"

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

def stochrsi_cross(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes)-period)]
    if len(rsi_vals) < 3:
        return None
    prev = rsi_vals[-3:-1]
    last = rsi_vals[-1]
    avg = np.mean(prev)
    if last > avg and last < 50:
        return 'bullish'
    elif last < avg and last > 50:
        return 'bearish'
    return None

def detect_volume_spike(volumes):
    avg = np.mean(volumes[:-1])
    return volumes[-1] > avg * VOLUME_SPIKE_RATIO

def detect_triangle_breakout(candles):
    highs = [float(c[2]) for c in candles[-10:]]
    lows = [float(c[3]) for c in candles[-10:]]
    closes = [float(c[4]) for c in candles[-10:]]
    upper = max(highs)
    lower = min(lows)
    range_pct = (upper - lower) / closes[-1]
    if range_pct < 0.02:
        if closes[-1] > upper:
            return '▲'
        elif closes[-1] < lower:
            return '▼'
    return ''

def detect_momentum(closes, volumes, triangle):
    ema_fast = calc_ema(closes[-10:], 9)
    ema_slow = calc_ema(closes[-10:], 21)
    rsi_val = rsi(closes)
    macd_val = macd(closes)
    vol_spike = detect_volume_spike(volumes)
    score = 0
    if ema_fast > ema_slow: score += 1
    if rsi_val > 50: score += 1
    if macd_val > 0: score += 1
    if vol_spike: score += 1
    if triangle: score += 1
    return score >= 3

async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]

            combined = []
            for tf in INDICATOR_TFS:
                ind = closes[tf]
                indicators = {
                    'EMA': calc_ema(ind, 9) > calc_ema(ind, 21),
                    'RSI': rsi(ind) > 50,
                    'MACD': macd(ind) > 0,
                    'SAR': psar(ind),
                    'BOLL': bollinger_band(ind),
                    'STOCH': stochrsi_cross(ind) == 'bullish'
                }
                combined.append(indicators)

            summary = {k: sum(ind[k] for ind in combined) for k in combined[0]}
            score = sum(v > 0 for v in summary.values())
            triangle = detect_triangle_breakout(candles[TRIANGLE_TF])
            momentum = detect_momentum(closes[MOMENTUM_TF], volumes, triangle)
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100
            vol_spike = detect_volume_spike(volumes)

            if score < MIN_SCORE or abs(price_change) < PRICE_CHANGE_THRESHOLD:
                return None

            trend = "bullish" if price_change > 0 else "bearish"
            label = "(Early)" if momentum else "(Confirmed)"
            indicators_fmt = " - ".join([f"{k}:{'S' if summary[k] else 'W'}" for k in summary])
            msg = f"**{symbol}** {triangle}{' (M)' if momentum else ''}{' Vol↑' if vol_spike else ''} | {price_change:+.2f}% | Score:{score} | {label} | {indicators_fmt}"
            return trend, label, score, abs(price_change), msg
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

def format_ranked_list(entries):
    return "\n\n".join([f"{i+1}. {entry[4]}" for i, entry in enumerate(entries)])

def format_report(bullish, bearish):
    msg = "*📊 Binance Futures Trend Scanner*\n\n"
    if bullish:
        msg += "🚀 *Bullish Signals:*\n" + format_ranked_list(bullish) + "\n\n"
    if bearish:
        msg += "🔻 *Bearish Signals:*\n" + format_ranked_list(bearish)
    return msg

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    results = await asyncio.gather(*tasks)
    filtered = [r for r in results if r]
    bullish = sorted([r for r in filtered if r[0] == "bullish"], key=lambda x: (-x[2], -x[3]))[:10]
    bearish = sorted([r for r in filtered if r[0] == "bearish"], key=lambda x: (-x[2], -x[3]))[:10]
    return bullish, bearish

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        bullish, bearish = await scan_market(session, symbols)
        if bullish or bearish:
            msg = format_report(bullish, bearish)
            await send_telegram(session, msg)
        else:
            await send_telegram(session, "🕵️‍♂️ No strong signals detected at this time.")
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
