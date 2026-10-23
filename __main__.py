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
import math
import time
from typing import Optional

import yfinance as yf

from train import train, timing
from agent import decision, newsfeed
from agent.agent import TickerAgent

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["NBIS", "AVGO", "GOOG"]  # the model is trained on exactly these tickers (and nothing else)

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
    print(f"\n{ticker} @ {time.strftime('%H:%M:%S')}  (features as of {result.get('as_of', 'n/a')})")
    print(f"   Headline : {' | '.join(result.get('headlines') or ['(none)'])[:100]}")
    print(f"   Sentiment: {result['sentiment_score']:.2f}")
    validated = "validated out-of-sample" if result.get("model_validated") else "NOT validated out-of-sample"
    bench = result.get("benchmark", "benchmark")
    print(f"   XGBoost Δ: {result['predicted_delta']:+.2f}% vs {bench} over {timing.BASE_HORIZON}d ({validated})")
    print(f"   VIX      : {ind.get('vix_current', 'N/A')}")
    print(f"   Pullback : {'Yes' if ind.get('pullback_buy_setup') else 'No'}")
    dte = ind.get("days_to_earn")
    if dte is not None and math.isfinite(dte):
        print(f"   Earnings : next in {dte:.0f} day(s)")

    m = result.get("manifest") or {}
    if m.get("reliable"):
        edge = "  (still rising at window edge: read as 'no earlier than')" if m["peak_at_window_edge"] else ""
        print(f"   ~80% by  : {m['reach_date']} (+{m['reach_horizon']}d, {m['direction']})")
        print(f"   Peak     : {m['peak_delta_pct']:+.2f}% on {m['peak_date']} (+{m['peak_horizon']}d){edge}")
    else:
        print(f"   Timing   : withheld — {m.get('reason', 'n/a')}")

    d = result.get("signal_detail") or {}
    if "mu_total_pct" in d:
        print(f"   Decision : E[excess move] {d['mu_total_pct']:+.2f}% (model {d['mu_model_pct']:+.2f}% + sentiment "
              f"{d['mu_sentiment_pct']:+.2f}%) ± {d['sigma_pct']:.2f}% | P(beats {d['cost_pct']}% costs) = {d['p_win']:.0%}")
        print(f"   Weights  : model beta {d['beta_eff']:.2f} (calibration {d['beta_raw']:.2f} × trust {d['trust_weight']:.2f}); "
              f"sentiment {d['gamma']:.3f}σ per +1.0 [{d['gamma_source']}]")
        if d.get("target_date"):
            print(f"   Review by: {d['target_date']}")
    else:
        print(f"   Decision : {d.get('reason', 'n/a')}")
    if "mu_total_pct" in d:
        print(f"   Signal   : **{result['signal']}** (Confidence: {result['signal_confidence']:.2f})")
    else:  # abstained before estimating anything (earnings blackout / no data): confidence is n/a, not 0
        print(f"   Signal   : **{result['signal']}** (abstained, no estimate made)")


async def consumer(id: int, ticker_agent: TickerAgent):
    while True:
        ticker = await QUEUE.get()
        try:
            print(f"Worker {id} started processing {ticker}")
            result = await ticker_agent.run(ticker)
            print_result(ticker, result)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
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
        print(f"[{ticker}] news fetch timed out after {NEWS_FETCH_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        print(f"[{ticker}] news fetch failed: {type(e).__name__}: {e}")
        return None

    if not items:
        print(f"📭 [{ticker}] yfinance returned no news items (yfinance {yf.__version__})")
        return None

    n = newsfeed.normalize(items[0])
    headline = n["title"] or n["summary"]
    if not headline:
        keys = list(items[0]) if isinstance(items[0], dict) else type(items[0]).__name__
        print(f"[{ticker}] newest news item has no title/summary; item keys: {keys}")
        return None
    return headline


async def news_source():
    print(f"📡 News director started — checking every {NEWS_POLL_INTERVAL_SEC}s")

    if RUN_ON_STARTUP:
        for ticker in TICKERS:
            headline = await latest_headline(ticker)
            if headline:
                LAST_HEADLINES[ticker] = headline  # so the first poll doesn't re-run the same news
            print(f"[startup] queueing {ticker} for an initial analysis")
            await QUEUE.put(ticker)

    while True:
        for ticker in TICKERS:
            headline = await latest_headline(ticker)
            if headline is None:
                continue  # reason already printed
            if headline == LAST_HEADLINES.get(ticker):
                print(f"[{ticker}] no new headline (latest: {headline[:60]!r})")
                continue
            print(f"📨 [NEW NEWS DETECTED] {ticker} → queued for processing headline: {headline[:50]} ...")
            LAST_HEADLINES[ticker] = headline
            await QUEUE.put(ticker)
        await asyncio.sleep(NEWS_POLL_INTERVAL_SEC)


def _report_task_death(task: asyncio.Task) -> None:
    # gather(..., return_exceptions=True) would otherwise swallow a crashed task silently
    if not task.cancelled() and task.exception() is not None:
        print(f"background task died: {task.exception()!r}")


async def main():
    # Training downloads ~10y for each ticker in TICKERS and runs walk-forward validation,
    # so keep it off the event loop. (Cached after the first run.)
    await asyncio.to_thread(train.xgboost, TICKERS, True)
    # Learn the LLM-sentiment weight from past logged calls whose outcomes have since played out.
    params = await asyncio.to_thread(decision.refit_sentiment)
    print(f"sentiment weight: {params['sentiment_gamma']:.3f}σ per +1.0 "
          f"({params['source']}, n={params['n']}, sd={params['sentiment_gamma_sd']:.3f})")
    ticker_agent = TickerAgent("gemini-3.8-flash", 0.0)

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