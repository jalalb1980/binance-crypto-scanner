import asyncio
import aiohttp
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
TELEGRAM_USER_ID = 'YOUR_USER_ID'
SLEEP_INTERVAL = 1800
MAX_CONCURRENT_REQUESTS = 50
MIN_SCORE_EARLY = 3
MIN_SCORE_CONFIRMED = 4
EARLY_MIN_PRICE_CHANGE = 3.0
CONFIRMED_MIN_PRICE_CHANGE = 10.0
VOLUME_SPIKE_RATIO = 2.0
CANDLE_LIMIT = 50

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
    try:
        async with session.get(url) as res:
            data = await res.json()
            return [s['symbol'] for s in data['symbols'] if is_futures_usdt(s)]
    except:
        return []

async def fetch_candles(session, symbol, interval):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={CANDLE_LIMIT}"
    async with session.get(url) as res:
        return await res.json()

async def send_telegram(session, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url, data={"chat_id": TELEGRAM_USER_ID, "text": text, "parse_mode": "Markdown"})

# === INDICATORS ===
def calc_ema(closes, span):
    return np.convolve(closes, np.ones(span)/span, mode='valid')[-1]

def rsi(closes, period=14):
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def macd(closes):
    ema12 = np.convolve(closes, np.ones(12)/12, mode='valid')
    ema26 = np.convolve(closes, np.ones(26)/26, mode='valid')
    macd_line = ema12[-len(ema26):] - ema26
    signal_line = np.convolve(macd_line, np.ones(9)/9, mode='valid')
    hist = macd_line[-len(signal_line):] - signal_line
    return hist

def psar(closes, af_step=0.02, af_max=0.2):
    psar = [closes[0]]
    ep = closes[0]
    af = af_step
    up = True
    for i in range(1, len(closes)):
        prev = psar[-1]
        if up:
            psar.append(prev + af * (ep - prev))
            if closes[i] > ep:
                ep = closes[i]
                af = min(af + af_step, af_max)
            elif closes[i] < psar[-1]:
                up = False
                psar[-1] = ep
                ep = closes[i]
                af = af_step
        else:
            psar.append(prev + af * (ep - prev))
            if closes[i] < ep:
                ep = closes[i]
                af = min(af + af_step, af_max)
            elif closes[i] > psar[-1]:
                up = True
                psar[-1] = ep
                ep = closes[i]
                af = af_step
    return psar

def bollinger_band(closes, period=20):
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    return closes[-1] > ma + std or closes[-1] < ma - std

def stochrsi(closes, period=14):
    rsi_vals = np.array([rsi(closes[i:i+period]) for i in range(len(closes)-period)])
    low = np.min(rsi_vals)
    high = np.max(rsi_vals)
    last = rsi_vals[-1] if len(rsi_vals) > 0 else 50
    return last > 80 or last < 20

def detect_momentum(closes):
    return closes[-1] > closes[-2] > closes[-3] or closes[-1] < closes[-2] < closes[-3]

def detect_triangle(candles):
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    upper = max(highs[-10:])
    lower = min(lows[-10:])
    width = upper - lower
    if width < 0.02 * closes[-1]:
        return '▲' if closes[-1] > closes[-2] else '▼'
    return '(Tx)'

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
                    'EMA': calc_ema(ind[-21:], 9) > calc_ema(ind[-21:], 21),
                    'RSI': rsi(ind) > 50,
                    'MACD': macd(ind)[-1] > 0,
                    'SAR': ind[-1] > psar(ind)[-1],
                    'BOLL': bollinger_band(ind),
                    'STOCH': stochrsi(ind)
                }
                combined.append(indicators)

            summary = {}
            for key in combined[0].keys():
                summary[key] = sum(1 for i in combined if i[key])

            score = sum(v > 0 for v in summary.values())
            momentum = detect_momentum(closes[MOMENTUM_TF])
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100
            triangle = detect_triangle(candles[TRIANGLE_TF])
            avg_vol = np.mean(volumes[:-1])
            vol_spike = volumes[-1] > avg_vol * VOLUME_SPIKE_RATIO

            label = None
            if score >= MIN_SCORE_CONFIRMED and abs(price_change) >= CONFIRMED_MIN_PRICE_CHANGE:
                label = "(Confirmed)"
            elif score == MIN_SCORE_EARLY and momentum and abs(price_change) >= EARLY_MIN_PRICE_CHANGE:
                label = "(Early)"
            if not label:
                return None

            trend = "bullish" if price_change > 0 else "bearish"
            indicators_fmt = " - ".join([f"{k}:{'S' if summary[k] else 'W'}" for k in summary])
            msg = f"**{symbol}** {triangle}{' (M)' if momentum else ''}{' Vol↑' if vol_spike else ''} | {price_change:+.2f}% | Score:{score} | {label} | {indicators_fmt}"
            return trend, label, score, abs(price_change), msg
        except:
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
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf = await scan_market(session, symbols)
        if bull_early or bull_conf or bear_early or bear_conf:
            msg = format_report(bull_early, bull_conf, bear_early, bear_conf)
            await send_telegram(session, msg)
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
