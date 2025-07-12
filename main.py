import asyncio
import aiohttp
import os
from datetime import datetime

# === ENV Variables ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # % movement threshold
INTERVAL = os.getenv("INTERVAL", "60")  # Bybit intervals: 1, 3, 5, 15, 30, 60, 120, etc.
SLEEP_INTERVAL = int(os.getenv("SLEEP_INTERVAL", "600"))  # 10 minutes in seconds

# === Telegram Sender ===
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

# === Symbol Fetching (Bybit) ===
async def fetch_symbols(session, url, filter_fn):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    async with session.get(url, headers=headers) as res:
        if res.status != 200:
            raise Exception(f"Symbol fetch error: {res.status}, message='{await res.text()}', url='{url}'")
        data = await res.json()
        return [s['symbol'] for s in data['result']['list'] if filter_fn(s)][:200]

def is_spot_usdt(s): return s['quoteCoin'] == 'USDT' and s['status'] == 'Trading'
def is_linear_usdt(s): return s['quoteCoin'] == 'USDT' and s['contractType'] == 'Linear'

# === Fetch Price Changes (Bybit) ===
async def fetch_change(session, symbol, category):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": 2
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        async with session.get(url, params=params, headers=headers) as res:
            data = await res.json()
            if data.get("retCode") != 0 or not data['result']['list']:
                return None
            candles = data['result']['list']
            old_price = float(candles[0][4])
            new_price = float(candles[1][4])
            if old_price > 0:
                change = ((new_price - old_price) / old_price) * 100
                msg = f"`{symbol}` {'UP' if change >= 0 else 'DOWN'} {abs(change):.2f}% | {INTERVAL}m: from {old_price:,.6f} → {new_price:,.6f}"
                if change >= THRESHOLD:
                    return ("gainer", change, f"🚀 {msg}")
                elif change <= -THRESHOLD:
                    return ("loser", abs(change), f"📉 {msg}")
    except Exception as e:
        print(f"⚠️ {symbol} error: {e}")
    return None

# === Group and Sort Movers ===
async def scan_market(session, symbols, category):
    tasks = [fetch_change(session, sym, category) for sym in symbols]
    results = await asyncio.gather(*tasks)

    gainers = sorted([r for r in results if r and r[0] == "gainer"], key=lambda x: x[1], reverse=True)
    losers = sorted([r for r in results if r and r[0] == "loser"], key=lambda x: x[1], reverse=True)

    return [g[2] for g in gainers], [l[2] for l in losers]

# === Full Scanner Execution ===
async def run_scan():
    async with aiohttp.ClientSession() as session:
        print(f"\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Starting scan...")

        try:
            spot_symbols, futures_symbols = await asyncio.gather(
                fetch_symbols(session, "https://api.bybit.com/v5/market/instruments-info?category=spot", is_spot_usdt),
                fetch_symbols(session, "https://api.bybit.com/v5/market/instruments-info?category=linear", is_linear_usdt)
            )
        except Exception as e:
            print(f"🚨 Symbol fetch error: {e}")
            return

        print(f"🔍 Spot symbols: {len(spot_symbols)}, Futures symbols: {len(futures_symbols)}")

        (spot_gainers, spot_losers), (futures_gainers, futures_losers) = await asyncio.gather(
            scan_market(session, spot_symbols, category="spot"),
            scan_market(session, futures_symbols, category="linear")
        )

        # === Format Spot Message ===
        if spot_gainers or spot_losers:
            message = f"📊 *Bybit Spot Movers (±{THRESHOLD}% in {INTERVAL}):*\n\n"
            if spot_gainers:
                message += "*🚀 Gainers:*\n" + "\n".join(spot_gainers) + "\n\n"
            if spot_losers:
                message += "*📉 Losers:*\n" + "\n".join(spot_losers)
            await send_telegram(session, message)
        else:
            print("✅ No Spot movers found.")

        # === Format Futures Message ===
        if futures_gainers or futures_losers:
            message = f"📈 *Bybit Futures Movers (±{THRESHOLD}% in {INTERVAL}):*\n\n"
            if futures_gainers:
                message += "*🚀 Gainers:*\n" + "\n".join(futures_gainers) + "\n\n"
            if futures_losers:
                message += "*📉 Losers:*\n" + "\n".join(futures_losers)
            await send_telegram(session, message)
        else:
            print("✅ No Futures movers found.")

# === Loop Forever ===
async def main():
    while True:
        try:
            await run_scan()
        except Exception as e:
            print("🚨 Error during scan:", e)
        print(f"✅ Sleeping for {SLEEP_INTERVAL // 60} minutes...\n")
        await asyncio.sleep(SLEEP_INTERVAL)

# === Start the Bot ===
if __name__ == "__main__":
    asyncio.run(main())
