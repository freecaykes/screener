# =============================================================================
# MULTI-TICKER ASYNCIO AGENT — LOW CPU / MEMORY (Recommended)
# =============================================================================
# pip install yfinance langgraph langchain langchain-openai pandas xgboost joblib
# (pandas_ta is no longer needed by the model path; pandas_market_calendars is optional
#  and gives exact NYSE holidays for the timing dates)
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
from typing import Optional

import yfinance as yf

from train import train, timing
from agent import newsfeed
from agent.agent import TickerAgent

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["AVGO"]  # the model is trained on exactly these tickers (and nothing else)

# =============================================================================
# CONCURRENT TICKER PROCESSOR (with semaphore)
# =============================================================================
QUEUE: asyncio.Queue = asyncio.Queue()
NEWS_POLL_INTERVAL_SEC = 300
LAST_HEADLINES: dict[str, str] = {}
MAX_CONCURRENT = 5
NEWS_FETCH_TIMEOUT_SEC = 30   # a hung yfinance call no longer stalls the loop silently
RUN_ON_STARTUP = True         # analyse each ticker once at startup, even if there is no news


def print_result(ticker: str, result: dict) -> None:
    ind = result["indicators"]
    print(f"\n🔥 {ticker} @ {time.strftime('%H:%M:%S')}  (features as of {result.get('as_of', 'n/a')})")
    print(f"   Headline : {' | '.join(result.get('headlines') or ['(none)'])[:100]}")
    print(f"   Sentiment: {result['sentiment_score']:.2f}")
    validated = "validated out-of-sample" if result.get("model_validated") else "NOT validated out-of-sample"
    print(f"   XGBoost Δ: {result['predicted_delta']:+.2f}% over {timing.BASE_HORIZON}d ({validated})")
    print(f"   VIX      : {ind.get('vix_current', 'N/A')}")
    print(f"   Pullback : {'Yes' if ind.get('pullback_buy_setup') else 'No'}")

    m = result.get("manifest") or {}
    if m.get("reliable"):
        edge = "  (still rising at window edge: read as 'no earlier than')" if m["peak_at_window_edge"] else ""
        print(f"   ~80% by  : {m['reach_date']} (+{m['reach_horizon']}d, {m['direction']})")
        print(f"   Peak     : {m['peak_delta_pct']:+.2f}% on {m['peak_date']} (+{m['peak_horizon']}d){edge}")
    else:
        print(f"   Timing   : withheld — {m.get('reason', 'n/a')}")

    print(f"   Signal   : **{result['signal']}** (Confidence: {result['signal_confidence']:.2f})")


async def consumer(id: int, ticker_agent: TickerAgent):
    while True:
        ticker = await QUEUE.get()
        try:
            print(f"🔧 Worker {id} started processing {ticker}")
            result = await ticker_agent.run(ticker)
            print_result(ticker, result)
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
        finally:
            QUEUE.task_done()


async def latest_headline(ticker: str) -> Optional[str]:
    """Newest headline for `ticker`, or None. Every failure path says why it returned None."""
    try:
        # NOTE: on timeout the worker thread keeps running in the background (threads
        # can't be killed), but the loop moves on instead of waiting forever.
        items = await asyncio.wait_for(
            asyncio.to_thread(newsfeed.fetch_news, ticker, 1), timeout=NEWS_FETCH_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        print(f"⏱️  [{ticker}] news fetch timed out after {NEWS_FETCH_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        print(f"⚠️  [{ticker}] news fetch failed: {type(e).__name__}: {e}")
        return None

    if not items:
        print(f"📭 [{ticker}] yfinance returned no news items (yfinance {yf.__version__})")
        return None

    n = newsfeed.normalize(items[0])
    headline = n["title"] or n["summary"]
    if not headline:
        keys = list(items[0]) if isinstance(items[0], dict) else type(items[0]).__name__
        print(f"❓ [{ticker}] newest news item has no title/summary; item keys: {keys}")
        return None
    return headline


async def news_source():
    print(f"📡 News director started — checking every {NEWS_POLL_INTERVAL_SEC}s")

    if RUN_ON_STARTUP:
        for ticker in TICKERS:
            headline = await latest_headline(ticker)
            if headline:
                LAST_HEADLINES[ticker] = headline  # so the first poll doesn't re-run the same news
            print(f"▶️  [startup] queueing {ticker} for an initial analysis")
            await QUEUE.put(ticker)

    while True:
        for ticker in TICKERS:
            headline = await latest_headline(ticker)
            if headline is None:
                continue  # reason already printed
            if headline == LAST_HEADLINES.get(ticker):
                print(f"💤 [{ticker}] no new headline (latest: {headline[:60]!r})")
                continue
            print(f"📨 [NEW NEWS DETECTED] {ticker} → queued for processing headline: {headline[:50]} ...")
            LAST_HEADLINES[ticker] = headline
            await QUEUE.put(ticker)
        await asyncio.sleep(NEWS_POLL_INTERVAL_SEC)


def _report_task_death(task: asyncio.Task) -> None:
    # gather(..., return_exceptions=True) would otherwise swallow a crashed task silently
    if not task.cancelled() and task.exception() is not None:
        print(f"💥 background task died: {task.exception()!r}")


async def main():
    # Training downloads ~10y for each ticker in TICKERS and runs walk-forward validation,
    # so keep it off the event loop. (Cached after the first run.)
    await asyncio.to_thread(train.xgboost, TICKERS)
    ticker_agent = TickerAgent("gemini-3.6-flash", 0.0)

    consumers = [
        asyncio.create_task(consumer(i, ticker_agent))
        for i in range(MAX_CONCURRENT)
    ]
    producer = asyncio.create_task(news_source())
    for t in [producer, *consumers]:
        t.add_done_callback(_report_task_death)

    await asyncio.gather(producer, *consumers, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Async agent shut down gracefully.")