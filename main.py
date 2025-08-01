import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = '7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ'
TELEGRAM_USER_ID = '7061959697'
MAX_CONCURRENT_REQUESTS = 50
CANDLE_LIMIT = 100
TIMEFRAMES = ["30m", "4h", "1d"]
LOW_TF, MID_TF, HIGH_TF = TIMEFRAMES
MOMENTUM_TF = LOW_TF
INDICATOR_TFS = [MID_TF, HIGH_TF]
TRIANGLE_TF = MID_TF
PRICE_TF = MID_TF
VOLUME_SPIKE_RATIO = 2.0
MIN_SCORE = 4
MIN_PRICE_CHANGE = 2.0

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
    closes = np.array(closes)
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

def macd_histogram(closes):
    ema12 = calc_ema(closes[-26:], 12)
    ema26 = calc_ema(closes[-26:], 26)
    macd = ema12 - ema26
    signal = calc_ema(closes[-9:], 9)
    return macd - signal

def psar_logic(closes):
    return closes[-1] > closes[-2]

def bollinger_break(closes, period=20):
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return closes[-1] > ma + std or closes[-1] < ma - std

def stochrsi_cross(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes)-period)]
    if len(rsi_vals) < 3:
        return False, False
    fast = rsi_vals[-1]
    slow = np.mean(rsi_vals[-3:])
    bullish = fast > slow and fast < 50
    bearish = fast < slow and fast > 50
    return bullish, bearish

def detect_volume_spike(volumes):
    return volumes[-1] > np.mean(volumes[-6:-1]) * VOLUME_SPIKE_RATIO

def detect_triangle(candles):
    highs = [float(c[2]) for c in candles[-10:]]
    lows = [float(c[3]) for c in candles[-10:]]
    closes = [float(c[4]) for c in candles[-10:]]
    width = max(highs) - min(lows)
    if width / closes[-1] < 0.02:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return ''

def momentum_score(closes, volumes):
    score = 0
    if calc_ema(closes[-10:], 9) > calc_ema(closes[-10:], 21): score += 1
    if rsi(closes) > 50: score += 1
    if macd_histogram(closes) > 0: score += 1
    if detect_volume_spike(volumes): score += 1
    return score

async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]
            price_change = (closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2] * 100
            triangle = detect_triangle(candles[TRIANGLE_TF])
            vol_spike = detect_volume_spike(volumes)

            combined = []
            for tf in INDICATOR_TFS:
                ind = closes[tf]
                st_bull, st_bear = stochrsi_cross(ind)
                indicators = {
                    'EMA': calc_ema(ind, 9) > calc_ema(ind, 21),
                    'RSI': rsi(ind) > 50,
                    'MACD': macd_histogram(ind) > 0,
                    'SAR': psar_logic(ind),
                    'BOLL': bollinger_break(ind),
                    'STOCH': st_bull if price_change > 0 else st_bear
                }
                combined.append(indicators)

            summary = {k: sum(1 for i in combined if i[k]) for k in combined[0]}
            indicator_score = sum(1 for v in summary.values() if v > 0)
            momentum = momentum_score(closes[MOMENTUM_TF], volumes)
            is_valid = indicator_score >= MIN_SCORE

            if not is_valid:
                return None

            trend = "bullish" if price_change > 0 else "bearish"
            label = "(Confirmed)" if momentum >= 3 else "(Early)"
            mark = triangle + (" (M)" if momentum >= 3 else "")
            indicators_fmt = " - ".join([f"{k}:{'S' if summary[k] else 'W'}" for k in summary])
            msg = f"**{symbol}** {mark}{' Vol↑' if vol_spike else ''} | {price_change:+.2f}% | Score:{indicator_score} | {label} | {indicators_fmt}"
            return trend, label, indicator_score, abs(price_change), msg
        except Exception as e:
            print(f"❌ {symbol} error: {e}")
            return None

def format_ranked_list(entries):
    return "\n\n".join([f"{i+1}. {entry[4]}" for i, entry in enumerate(entries)])

def format_report(bull_early, bull_conf, bear_early, bear_conf):
    msg = "*📊 Binance Futures Trend Scanner*\n\n"
    if bull_conf or bull_early:
        msg += "🚀 *Bullish Signals:*\n"
        if bull_conf: msg += "🟢 *Confirmed:*\n" + format_ranked_list(bull_conf) + "\n\n"
        if bull_early: msg += "🟡 *Early:*\n" + format_ranked_list(bull_early) + "\n\n"
    if bear_conf or bear_early:
        msg += "🔻 *Bearish Signals:*\n"
        if bear_conf: msg += "🔴 *Confirmed:*\n" + format_ranked_list(bear_conf) + "\n\n"
        if bear_early: msg += "🟠 *Early:*\n" + format_ranked_list(bear_early)
    return msg

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    results = await asyncio.gather(*tasks)
    filtered = [r for r in results if r]

    def sort_group(trend, label):
        return sorted(
            [r for r in filtered if r[0] == trend and r[1] == label],
            key=lambda x: (-x[2], -x[3])
        )[:8]

    return sort_group("bullish", "(Early)"), sort_group("bullish", "(Confirmed)"), sort_group("bearish", "(Early)"), sort_group("bearish", "(Confirmed)")

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        b1, b2, s1, s2 = await scan_market(session, symbols)
        if b1 or b2 or s1 or s2:
            msg = format_report(b1, b2, s1, s2)
            await send_telegram(session, msg)
        else:
            await send_telegram(session, "📉 *No coins matched the trend filters at this time.*")
        print("✅ Scan finished.\n")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(run_scan())
