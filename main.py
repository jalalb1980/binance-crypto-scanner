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
STOCHRSI_TF = HIGH_TF

# === UTILITIES ===
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

def detect_momentum(closes, volumes):
    ema_fast = calc_ema(closes[-10:], 9)
    ema_slow = calc_ema(closes[-10:], 21)
    rsi_val = rsi(closes)
    macd = macd_hist(closes)
    vol_spike = volumes[-1] > np.mean(volumes[:-1]) * VOLUME_SPIKE_RATIO
    score = sum([ema_fast > ema_slow, rsi_val > 50, macd > 0, vol_spike])
    return score, vol_spike

def get_trend_direction(closes):
    ema_fast = calc_ema(closes, 9)
    ema_slow = calc_ema(closes, 21)
    if ema_fast > ema_slow:
        return 'bullish'
    elif ema_fast < ema_slow:
        return 'bearish'
    return 'neutral'

def stochrsi_kd(closes, period=14):
    rsi_vals = [rsi(closes[i:i+period]) for i in range(len(closes) - period)]
    if len(rsi_vals) < 5:
        return None, None
    k = rsi_vals[-1]
    d = np.mean(rsi_vals[-3:])
    return k, d

async def analyze_symbol(session, symbol, semaphore):
    async with semaphore:
        try:
            tfs = set(INDICATOR_TFS + [MOMENTUM_TF, PRICE_TF, TRIANGLE_TF, STOCHRSI_TF])
            candles = {tf: await fetch_candles(session, symbol, tf) for tf in tfs}
            closes = {tf: [float(c[4]) for c in candles[tf]] for tf in tfs}
            volumes = [float(c[5]) for c in candles[MID_TF]]
            last_price = closes[PRICE_TF][-1]
            price_change = ((closes[PRICE_TF][-1] - closes[PRICE_TF][-2]) / closes[PRICE_TF][-2]) * 100
            triangle = detect_triangle(candles[TRIANGLE_TF])
            indicator_score = sum([
                calc_ema(closes[MID_TF], 9) > calc_ema(closes[MID_TF], 21),
                rsi(closes[MID_TF]) > 50,
                macd_hist(closes[MID_TF]) > 0,
                psar(closes[MID_TF]),
                bollinger_band(closes[MID_TF]),
            ])
            momentum_score, _ = detect_momentum(closes[MOMENTUM_TF], volumes)

            label = None
            classification = None
            if indicator_score >= MIN_SCORE_CONFIRMED and abs(price_change) >= PRICE_CHANGE_THRESHOLD:
                label = "(Confirmed)"
                classification = f"{get_trend_direction(closes[HIGH_TF]).capitalize()} (M)" if momentum_score >= 3 else get_trend_direction(closes[HIGH_TF]).capitalize()
            elif indicator_score >= MIN_SCORE_EARLY and momentum_score >= 3 and abs(price_change) >= 2:
                label = "(Early)"
                classification = f"{get_trend_direction(closes[HIGH_TF]).capitalize()} (M)"

            indicators_fmt = f"EMA:{'S' if calc_ema(closes[MID_TF], 9) > calc_ema(closes[MID_TF], 21) else 'W'} - RSI:{'S' if rsi(closes[MID_TF]) > 50 else 'W'} - MACD:{'S' if macd_hist(closes[MID_TF]) > 0 else 'W'} - SAR:{'S' if psar(closes[MID_TF]) else 'W'} - BOLL:{'S' if bollinger_band(closes[MID_TF]) else 'W'}"

            msg = f"**{symbol}** {triangle} | {price_change:+.2f}% | Price: {last_price:.2f} | Score:{indicator_score} | {label} | *{classification}* | {indicators_fmt}" if label and classification else None

            # --- StochRSI cross detection ---
            k, d = stochrsi_kd(closes[STOCHRSI_TF])
            stoch_signal = None
            if k and d:
                if k > d and k < 40:
                    stoch_signal = ("bullish", price_change, f"**{symbol}** | {price_change:+.2f}% | K:{k:.2f} > D:{d:.2f}")
                elif k < d and k > 60:
                    stoch_signal = ("bearish", price_change, f"**{symbol}** | {price_change:+.2f}% | K:{k:.2f} < D:{d:.2f}")

            return (get_trend_direction(closes[HIGH_TF]), label, indicator_score, abs(price_change), msg), stoch_signal
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None, None

def format_ranked_list(entries):
    return "\n\n".join([f"{i+1}. {entry[4]}" for i, entry in enumerate(entries)])

def format_stochrsi_list(entries):
    return "\n\n".join([f"{i+1}. {entry[2]}" for i, entry in enumerate(entries)])

def format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear):
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
    if stoch_bull or stoch_bear:
        msg += "📈 *StochRSI Signals:*\n"
        if stoch_bull:
            msg += "🔼 *Bullish:* \n" + format_stochrsi_list(stoch_bull) + "\n\n"
        if stoch_bear:
            msg += "🔽 *Bearish:* \n" + format_stochrsi_list(stoch_bear)
    return msg

async def scan_market(session, symbols):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [analyze_symbol(session, s, semaphore) for s in symbols]
    results = await asyncio.gather(*tasks)
    main_signals = [r[0] for r in results if r[0]]
    stoch_signals = [r[1] for r in results if r[1]]

    def filter_and_sort(trend, label):
        return sorted([r for r in main_signals if r[0] == trend and r[1] == label], key=lambda x: (-x[2], -x[3]))[:10]

    bull_conf = filter_and_sort("bullish", "(Confirmed)")
    bull_early = filter_and_sort("bullish", "(Early)")
    bear_conf = filter_and_sort("bearish", "(Confirmed)")
    bear_early = filter_and_sort("bearish", "(Early)")

    stoch_bull = sorted([s for s in stoch_signals if s[0] == "bullish"], key=lambda x: -x[1])[:10]
    stoch_bear = sorted([s for s in stoch_signals if s[0] == "bearish"], key=lambda x: x[1])[:10]

    return bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear

async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n⏱️ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Scanning...")
        symbols = await fetch_symbols(session)
        bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear = await scan_market(session, symbols)
        if any([bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear]):
            msg = format_report(bull_early, bull_conf, bear_early, bear_conf, stoch_bull, stoch_bear)
            await send_telegram(session, msg)
        else:
            await send_telegram(session, "*📊 No coins match filter criteria at the moment.*")
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
