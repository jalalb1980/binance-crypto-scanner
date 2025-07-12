import asyncio
import aiohttp
import os
from datetime import datetime

# === ENV Variables ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
THRESHOLD = os.getenv("THRESHOLD")  # % movement threshold
INTERVAL = os.getenv("INTERVAL")  # Bybit intervals: 1, 3, 5, 15, 30, 60, 120, etc.
SLEEP_INTERVAL = os.getenv("SLEEP_INTERVAL")  # 10 minutes in seconds

# === PROXY SETUP ===
PROXY_BASE = "https://workers-playground-summer-heart-2306.jalal-binche.workers.dev"
SPOT_INFO = f"{PROXY_BASE}/v5/market/instruments-info?category=spot"
FUTURES_INFO = f"{PROXY_BASE}/v5/market/instruments-info?category=linear"
KLINE_URL = f"{PROXY_BASE}/v5/market/kline"

# === TELEGRAM NOTIFIER ===
async def send_telegram(session, message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_USER_ID, "text": message}
    async with session.post(url, data=payload) as res:
        if res.status != 200:
            print("❌ Telegram error:", await res.text())

# === SYMBOL FETCHING ===
async def fetch_symbols(session, url):
    try:
        async with session.get(url) as res:
            data = await res.json()
            return [i['symbol'] for i in data.get('result', {}).get('list', [])]
    except Exception as e:
        print(f"❌ Symbol fetch error: {e}")
        return []

# === PRICE CHANGE SCANNER ===
async def fetch_change(session, symbol, category):
    params = {
        "category": category,
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": 2
    }
    try:
        async with session.get(KLINE_URL, params=params) as res:
            data = await res.json()
            kline = data.get('result', {}).get('list', [])
            if isinstance(kline, list) and len(kline) == 2:
                old_price = float(kline[0][4])
                new_price = float(kline[1][4])
                if old_price > 0:
                    change = ((new_price - old_price) / old_price) * 100
                    msg = f"{symbol}: {'UP' if change >= 0 else 'DOWN'} {abs(change):.2f}% | {INTERVAL}m: {old_price:.4f} → {new_price:.4f}"
                    if change >= THRESHOLD:
                        return f"🚀 {msg}"
                    elif change <= -THRESHOLD:
                        return f"📉 {msg}"
    except Exception as e:
        print(f"⚠️ {symbol} fetch error: {e}")
    return None

# === MARKET SCAN ===
async def scan(session, symbols, category):
    tasks = [fetch_change(session, sym, category) for sym in symbols]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]

# === MAIN SCAN LOOP ===
async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Starting scan...")

        spot_symbols, futures_symbols = await asyncio.gather(
            fetch_symbols(session, SPOT_INFO),
            fetch_symbols(session, FUTURES_INFO)
        )

        print(f"✅ Spot: {len(spot_symbols)} symbols, Futures: {len(futures_symbols)} symbols")

        spot_changes, futures_changes = await asyncio.gather(
            scan(session, spot_symbols, "spot"),
            scan(session, futures_symbols, "linear")
        )

        if spot_changes:
            msg = f"📊 Spot Movers (±{THRESHOLD}% in {INTERVAL}m):\n" + "\n".join(spot_changes)
            await send_telegram(session, msg)
        else:
            print("✅ No Spot movers found.")

        if futures_changes:
            msg = f"📈 Futures Movers (±{THRESHOLD}% in {INTERVAL}m):\n" + "\n".join(futures_changes)
            await send_telegram(session, msg)
        else:
            print("✅ No Futures movers found.")

# === LOOP FOREVER ===
async def main():
    while True:
        try:
            await run_scan()
        except Exception as e:
            print("🚨 Scan error:", e)
        print(f"🛌 Sleeping for {SLEEP_INTERVAL // 60} minutes...\n")
        await asyncio.sleep(SLEEP_INTERVAL)

# === START ===
if __name__ == "__main__":
    asyncio.run(main())
