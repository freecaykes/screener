# =============================================================================
# MULTI-TICKER ASYNCIO AGENT — LOW CPU / MEMORY (Recommended)
# =============================================================================
# pip install yfinance langgraph langchain langchain-openai pandas pandas_ta xgboost joblib
#
# This version uses:
#   • asyncio + Semaphore (max 5 concurrent analyses)
#   • No per-ticker threads
#   • Blocking calls wrapped in asyncio.to_thread
#   • Async LLM calls
#   • In-memory cache for last headline (avoids duplicate work)
#
# Much lighter on CPU & memory than threading!
# =============================================================================

import asyncio
import time

import yfinance as yf

import train.train
from agent.agent import TickerAgent

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["NVDA"]   # ← Add/remove anytime

# =============================================================================
# CONCURRENT TICKER PROCESSOR (with semaphore)
# =============================================================================
QUEUE: asyncio.Queue = asyncio.Queue()
NEWS_POLL_INTERVAL_SEC = 5
LAST_HEADLINES: dict[str, str] = {}
MAX_CONCURRENT = 5

async def consumer(id: int, ticker_agent: TickerAgent):

    while True:
        ticker = await QUEUE.get()
        start = time.time()
        try:
            print(f"🔧 Worker {id} started processing {ticker}")
            result = await ticker_agent.run(ticker)

            print(f"✅ PROCESSED {ticker} | "
                  f"Sentiment: {result['sentiment_score']:.2f} | "
                  f"XGBoost Δ: {result['predicted_delta']:+.2f}% "
                  f"({time.time()-start:.1f}s)")
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
        finally:
            QUEUE.task_done()

async def news_source():
    global QUEUE
    print(f"📡 News director started — checking every {NEWS_POLL_INTERVAL_SEC}s")
    while True:
        for ticker in TICKERS:
            try:
                news = await asyncio.to_thread(lambda t=ticker: yf.Ticker(t).news[:1])
                if not news:
                    continue
                headline = news[0].get("content", "").get("title") or news[0].get("content", "").get("description", "")
                if headline and headline != LAST_HEADLINES.get(ticker, ""):
                    print(f"📨 [NEW NEWS DETECTED] {ticker} → queued for processing headline: {headline[:50]} ...")
                    await QUEUE.put(ticker)
                    LAST_HEADLINES[ticker] = headline
            except Exception as e:
                print(e)
                pass  # silent on transient network issues
        await asyncio.sleep(NEWS_POLL_INTERVAL_SEC)


async def main():
    train.xgboost(TICKERS)
    ticker_agent =  TickerAgent("gemini-2.5-flash", 0.0)

    consumers = [
        asyncio.create_task(consumer(i, ticker_agent))
        for i in range(MAX_CONCURRENT)
    ]

    producer = asyncio.create_task(news_source())

    await asyncio.gather(producer, *consumers, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Async agent shut down gracefully.")