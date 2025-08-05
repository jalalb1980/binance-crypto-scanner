import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
MAX_CONCURRENT_REQUESTS = 50
CANDLE_LIMIT = 100
TIMEFRAME = "30m"

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

def macd_data(closes):
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    return macd_line, signal_line

def ema(values, span):
    weights = np.exp(np.linspace(-1., 0., span))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    return a[-1]

def stochrsi_cross(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes)-period)]
    if len(rsi_vals) < 3:
        return None
    k_line = np.array(rsi_vals[-3:])
    d_line = np.convolve(k_line, np.ones(3)/3, mode='valid')
    if len(d_line) < 2:
        return None
    # Bullish crossover: K crosses above D from oversold
    if k_line[-2] < d_line[-1] and k_line[-1] > d_line[-1] and k_line[-1] < 20:
        return 'Bullish'
    # Bearish crossover: K crosses below D from overbought
    if k_line[-2] > d_line[-1] and k_line[-1] < d_line[-1] and k_line[-1] > 80:
        return 'Bearish'
    return None

async def analyze_stochrsi(session, symbol, semaphore):
    async with semaphore:
        try:
            candles = await fetch_candles(session, symbol, TIMEFRAME)
            closes = np.array([float(c[4]) for c in candles])
            stoch_signal = stochrsi_cross(closes)
            if not stoch_signal:
                return None
            macd_line, signal_line = macd_data(closes)
            macd_valid = macd_line > 0 and signal_line > 0 if stoch_signal == 'Bullish' else macd_line < 0 and signal_line < 0
            if not macd_valid:
                return None
            macd_strength = abs(macd_line - signal_line)
            return symbol, stoch_signal, macd_strength
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

def format_stochrsi_list(coins):
    if not coins:
        return ""
    msg = "*🧠 StochRSI Cross Signals (MACD Confirmed)*\n"
    bullish = [c for c in coins if c[1] == 'Bullish']
    bearish = [c for c in coins if c[1] == 'Bearish']

    if bullish:
        msg += "\n📈 *Uptrend Candidates:*\n"
        for i, c in enumerate(sorted(bullish, key=lambda x: -x[2])[:10]):
            msg += f"{i+1}. `{c[0]}` | MACD Δ: `{c[2]:.4f}`\n"
    if bearish:
        msg += "\n📉 *Downtrend Candidates:*\n"
        for i, c in enumerate(sorted(bearish, key=lambda x: -x[2])[:10]):
            msg += f"{i+1}. `{c[0]}` | MACD Δ: `{c[2]:.4f}`\n"

    return msg

async def run_stochrsi_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning StochRSI Crossovers...")
        symbols = await fetch_symbols(session)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [analyze_stochrsi(session, s, semaphore) for s in symbols]
        results = await asyncio.gather(*tasks)
        final = [r for r in results if r]
        message = format_stochrsi_list(final)
        if message:
            await send_telegram(session, message)
        else:
            await send_telegram(session, "*🧠 StochRSI scan completed. No valid signals found.*")
        print("✅ Done.\n")

# === RUN ===
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(run_stochrsi_scan())
