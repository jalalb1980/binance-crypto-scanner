import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
CANDLE_LIMIT = 50
MAX_CONCURRENT_REQUESTS = 50

# Timeframes
TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
PRICE_TF = HIGH_TF

# Thresholds
MIN_SCORE_EARLY = 4
MIN_SCORE_CONFIRMED = 5
PRICE_CHANGE_THRESHOLD = 2.0
VOLUME_SPIKE_RATIO = 2.0

# === UTILITY ===
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

def stochrsi(closes, period=14):
    try:
        rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes)-period)]
        if len(rsi_vals) < 3:
            return False
        last_rsi = rsi_vals[-1]
        avg_rsi = np.mean(rsi_vals[-3:])
        if last_rsi > avg_rsi and last_rsi < 50:
            return True  # Bullish cross
        if last_rsi < avg_rsi and last_rsi > 50:
            return True  # Bearish cross
        return False
    except:
        return False

def detect_momentum(closes, volumes):
    try:
        ema_fast = calc_ema(closes[-10:], 9)
        ema_slow = calc_ema(closes[-10:], 21)
        rsi_val = rsi(closes)
        macd_hist = macd(closes)
        vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO
        return ema_fast > ema_slow and rsi_val > 50 and macd_hist > 0 and vol_spike
    except:
        return False

# === ANALYSIS ===
async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]

            combined = []
            for tf in INDICATOR_TFS:
                data = closes[tf]
                indicators = {
                    'EMA': calc_ema(data, 9) > calc_ema(data, 21),
                    'RSI': rsi(data) > 50,
                    'MACD': macd(data) > 0,
                    'SAR': psar(data),
                    'BOLL': bollinger_band(data),
                    'STOCHRSI': stochrsi(data)
                }
                combined.append(indicators)

            summary = {}
            for key in combined[0].keys():
                summary[key] = sum(1 for i in combined if i[key])

            score = sum(v > 0 for v in summary.values())
            momentum = detect_momentum(closes[MOMENTUM_TF], volumes)
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100

            if abs(price_change) < PRICE_CHANGE_THRESHOLD or score < 3:
                return None

            trend = "bullish" if price_change > 0 else "bearish"
            label = None
            if score >= MIN_SCORE_CONFIRMED:
                label = "(Confirmed)"
            elif score >= MIN_SCORE_EARLY and momentum:
                label = "(Early)"
            else:
                return None

            indicators_fmt = " - ".join([f"{k}:{'S' if summary[k] else 'W'}" for k in summary])
            msg = f"**{symbol}** {'(M)' if momentum else ''} | {price_change:+.2f}% | Score:{score} | {label} | {indicators_fmt}"
            return trend, label, score, abs(price_change), msg
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

# === SCANNER ===
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

        if bull_early or bull_conf or bear_early or bear_conf:
            msg = format_report(bull_early, bull_conf, bear_early, bear_conf)
        else:
            msg = "📉 *No coins matched the filter in this scan.*"

        await send_telegram(session, msg)
        print("✅ Scan finished.\n")

# === ENTRY ===
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(run_scan())
