import asyncio
import aiohttp
import os
from datetime import datetime

# === Use ENV variables for security ===
TELEGRAM_BOT_TOKEN = os.getenv("7993511855:AAFRUpzz88JsYflrqFIbv8OlmFiNnMJ_kaQ")
TELEGRAM_USER_ID = os.getenv("7061959697")  # should be numeric string

THRESHOLD = 10  # % movement threshold
INTERVAL = '1h'  # Binance interval (use stable ones: 1h, 2h, 4h, etc.)
SLEEP_INTERVAL = 600  # 10 minutes in seconds

# === Telegram Sender (auto-split messages) ===
async def send_telegram(session, message):
    MAX_LEN = 4000
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i+MAX_LEN] for i in range(0, len(message), MAX_LEN)]

    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_USER_ID,
            "text": chunk
        }
        async with session.post(url, data=payload) as res:
            if res.status != 200:
                print("❌ Telegram error:", await res.text())

# === Symbol Fetching ===
async def fetch_symbols(session, url, filter_fn):
    async with session.get(url) as res:
        data = await res.json()
        return [s['symbol'] for s in data['symbols'] if filter_fn(s)][:200]

def is_spot_usdt(s): return s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
def is_futures_usdt(s): return s['quoteAsset'] == 'USDT' and s.get('contractType') == 'PERPETUAL'

# === Fetch Price Changes ===
async def fetch_change(session, symbol, is_futures):
    base = "https://fapi.binance.com" if is_futures else "https://api.binance.com"
    path = "/fapi/v1/klines" if is_futures else "/api/v3/klines"
    url = f"{base}{path}"
    params = {'symbol': symbol, 'interval': INTERVAL, 'limit': 2}
    try:
        async with session.get(url, params=params) as res:
            data = await res.json()
            if isinstance(data, list) and len(data) == 2:
                old_price = float(data[0][4])
                new_price = float(data[1][4])
                if old_price > 0:
                    change = ((new_price - old_price) / old_price) * 100
                    msg = f"{symbol} {'UP' if change >= 0 else 'DOWN'} {abs(change):.2f}% | {INTERVAL}: from {old_price:,.6f} → {new_price:,.6f}"
                    if change >= THRESHOLD:
                        return ("gainer", change, f"🚀 {msg}")
                    elif change <= -THRESHOLD:
                        return ("loser", abs(change), f"📉 {msg}")
    except Exception as e:
        print(f"⚠️ {symbol} error: {e}")
    return None

# === Scan Market (sort results) ===
async def scan_market(session, symbols, is_futures):
    tasks = [fetch_change(session, sym, is_futures) for sym in symbols]
    results = await asyncio.gather(*tasks)

    gainers = sorted(
        [r for r in results if r and r[0] == "gainer"],
        key=lambda x: x[1],
        reverse=True
    )
    losers = sorted(
        [r for r in results if r and r[0] == "loser"],
        key=lambda x: x[1],
        reverse=True
    )

    return [g[2] for g in gainers], [l[2] for l in losers]

# === Scan and Send ===
async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Starting scan...")

        spot_symbols, futures_symbols = await asyncio.gather(
            fetch_symbols(session, "https://api.binance.com/api/v3/exchangeInfo", is_spot_usdt),
            fetch_symbols(session, "https://fapi.binance.com/fapi/v1/exchangeInfo", is_futures_usdt)
        )

        (spot_gainers, spot_losers), (futures_gainers, futures_losers) = await asyncio.gather(
            scan_market(session, spot_symbols, is_futures=False),
            scan_market(session, futures_symbols, is_futures=True)
        )

        if spot_gainers or spot_losers:
            message = f"📊 Spot Movers (±{THRESHOLD}% in {INTERVAL}):\n\n"
            if spot_gainers:
                message += "🚀 Gainers:\n" + "\n".join(spot_gainers) + "\n\n"
            if spot_losers:
                message += "📉 Losers:\n" + "\n".join(spot_losers)
            await send_telegram(session, message)
        else:
            print("✅ No Spot movers found.")

        if futures_gainers or futures_losers:
            message = f"📈 Futures Movers (±{THRESHOLD}% in {INTERVAL}):\n\n"
            if futures_gainers:
                message += "🚀 Gainers:\n" + "\n".join(futures_gainers) + "\n\n"
            if futures_losers:
                message += "📉 Losers:\n" + "\n".join(futures_losers)
            await send_telegram(session, message)
        else:
            print("✅ No Futures movers found.")

# === Main Loop ===
async def main():
    while True:
        try:
            await run_scan()
        except Exception as e:
            print("🚨 Error during scan:", e)
        print(f"✅ Sleeping for {SLEEP_INTERVAL // 60} minutes...\n")
        await asyncio.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
