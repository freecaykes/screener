# =============================================================================
# SWING-SETUP AGENT (asyncio, low CPU / memory)
# =============================================================================
# pip install yfinance langgraph langchain langchain-openai pandas xgboost joblib
# (pandas_market_calendars is optional: exact NYSE holidays for the time-stop date)
#
# What it does for each ticker in TICKERS (and nothing else is ever trained on):
#   1. after the close it looks for one of four swing setups on the last completed session:
#        pullback_buy  : dip in an uptrend that has just shown a reversal candle (~ near a local low)
#        breakout_buy  : close above the prior 20-day high on heavy volume
#        rally_sell / breakdown_sell : the mirror images
#   2. estimates the chance the trade reaches its profit target before its stop (train/setup_model.py)
#   3. has the LLM read the headlines IN THE CONTEXT of that setup (noise vs lasting damage / real catalyst)
#   4. prints BUY / SELL with an entry reference, stop, target and time stop -- or HOLD with the reason
#
# Triggers: a once-a-day scan after the close, new headlines, and once at startup.
# =============================================================================
import asyncio
import math
import os
import time
from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import logging

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated*",
    module=r"yfinance",
)

import yfinance as yf


from train import train
from agent import decision, newsfeed,TickerAgent
from config import get_config

cfg = get_config()

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["META", "NVDA"]  # your hand-picked list: the model learns from exactly these tickers (and nothing else)

# =============================================================================
# CONCURRENT TICKER PROCESSOR (with semaphore)
# =============================================================================
QUEUE: asyncio.Queue = asyncio.Queue()
NEWS_POLL_INTERVAL_SEC = 300
LAST_HEADLINES: dict[str, str] = {}
MAX_CONCURRENT = 5
NEWS_FETCH_TIMEOUT_SEC = 30   # a hung yfinance call no longer stalls the loop silently
SCAN_TIME = dtime(16, 10)     # New York time: daily scan, just after the close (bars are complete by 16:05)
_NY = ZoneInfo("America/New_York")


def seconds_until_next_scan(now: Optional[datetime] = None) -> float:
    """Seconds until the next weekday SCAN_TIME (New York). Holidays still scan; they just see no new bar."""
    now = (now or datetime.now(_NY)).astimezone(_NY)
    target = datetime.combine(now.date(), SCAN_TIME, tzinfo=_NY)
    if now >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def print_result(ticker: str, result: dict) -> None:
    ind = result.get("indicators") or {}
    d = result.get("signal_detail") or {}
    logging.info(f"{ticker} @ {time.strftime('%H:%M:%S')}  (last completed session: {result.get('as_of', 'n/a')}, "
          "vs {result.get('benchmark', 'benchmark')})")
    logging.info(f"   Headline : {' | '.join(result.get('headlines') or ['(none)'])[:100]}")
    dte = ind.get("days_to_earn")
    if dte is not None and math.isfinite(dte):
        logging.info(f"   Earnings : next in {dte:.0f} day(s)")

    setup = result.get("setup") or ""
    if not setup:
        if d.get("reason", "").startswith("no usable data"):
            logging.info(f"   Status   : HOLD -- {d['reason']}")
        else:
            logging.info(f"   Setup    : none active -> HOLD")
        return
    if setup in ("breakout_buy", "breakdown_sell"):
        what = ("closed {ind.get('bo_atr', float('nan')):.1f} ATR through the 20-day {'high' if setup == 'breakout_buy' else 'low'} "
                "on {ind.get('rel_volume', float('nan')):.1f}x volume")
    else:
        what = ("{'dip' if setup == 'pullback_buy' else 'rally'} of {ind.get('depth_atr', float('nan')):.1f} ATR from the 10-day "
                "{'high' if setup == 'pullback_buy' else 'low'}, reversal candle today")
    logging.info(f"   Setup    : {setup.upper()}  ({what}; ATR {ind.get('atr_pct', float('nan')):.2f}% of price)")

    news = d.get("news") or result.get("news_analysis")
    if news:
        logging.info(f"   News     : sentiment {news['sentiment']:+.2f}, thesis '{news['thesis']}', materiality {news['materiality']:.2f}"
              " -- {news.get('reason', '')}")
    if "p_final" in d:
        logging.info(f"   Odds     : any-day {d['p_any']:.0%} -> this setup {d['p_setup']:.0%} -> final {d['p_final']:.0%} "
              "(setup evidence {d['setup_trust']:.2f}, model {d['model_trust']:.2f}, news weight {d['gamma']:.2f} [{d['gamma_source']}])")
        logging.info(f"   Value    : EV {d['ev_R']:+.2f}R per trade after {d['cost_R']:.2f}R costs  "
              "({'validated' if d['validated'] else 'NOT validated out-of-sample'})")
    plan = d.get("plan")
    if plan:
        verb = "Buy" if d.get("direction", 1) > 0 else "Sell/short"
        logging.info(f"   Plan     : {verb} near {plan['entry_ref']:.2f} | stop {plan['stop']:.2f} ({plan['stop_pct']:.1f}%) | "
              "target {plan['target']:.2f} ({plan['target_pct']:.1f}%) | exit by {plan['max_hold_date']}")
    if result["signal"] == "HOLD":
        tag = "WATCH" if d.get("watch") else "HOLD"
        logging.info(f"   Signal   : **HOLD** [{tag}] {d.get('reason', '')}")
    else:
        logging.info(f"   Signal   : **{result['signal']}** (P(target before stop) {result['signal_confidence']:.0%})")


async def consumer(id: int, ticker_agent: TickerAgent):
    while True:
        ticker = await QUEUE.get()
        try:
            logging.info(f"🔧 Worker {id} started processing {ticker}")
            result = await ticker_agent.run(ticker)
            print_result(ticker, result)
        except Exception as e:
            logging.info(f"❌ Error processing {ticker}: {e}")
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
        logging.info(f"[{ticker}] news fetch timed out after {NEWS_FETCH_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        logging.info(f"[{ticker}] news fetch failed: {type(e).__name__}: {e}")
        return None

    if not items:
        logging.info(f"[{ticker}] yfinance returned no news items (yfinance {yf.__version__})")
        return None

    n = newsfeed.normalize(items[0])
    headline = n["title"] or n["summary"]
    if not headline:
        keys = list(items[0]) if isinstance(items[0], dict) else type(items[0]).__name__
        logging.info(f"[{ticker}] newest news item has no title/summary; item keys: {keys}")
        return None
    return headline


async def news_source():
    logging.info(f"News director started — checking every {NEWS_POLL_INTERVAL_SEC}s")

    while True:
        for ticker in cfg.general.tickers:
            headline = await latest_headline(ticker)
            if not headline:
                logging.info(f"[{ticker}] no new headline) running on LLM-found headline")
            elif headline == LAST_HEADLINES.get(ticker):
                logging.info(f"[{ticker}] no new headline (latest: {headline[:60]!r}) running on LLM-found headline")
            else:
                logging.info(f"[NEW NEWS DETECTED] {ticker} → queued for processing headline: {headline[:50]} ...")
                LAST_HEADLINES[ticker] = headline
                
            await QUEUE.put(ticker)
        await asyncio.sleep(NEWS_POLL_INTERVAL_SEC)


async def daily_scan():
    """Queue every ticker once a day just after the close, when the day's bar is complete."""
    while True:
        wait = seconds_until_next_scan()
        logging.info(f"🗓️  next daily scan in {wait/3600:.1f}h")
        await asyncio.sleep(wait)
        for ticker in cfg.general.tickers:
            logging.info(f"🗓️  [daily scan] queueing {ticker}")
            await QUEUE.put(ticker)
        await asyncio.sleep(60)  # don't fire twice for the same minute


def _report_task_death(task: asyncio.Task) -> None:
    # gather(..., return_exceptions=True) would otherwise swallow a crashed task silently
    if not task.cancelled() and task.exception() is not None:
        logging.info(f"💥 background task died: {task.exception()!r}")


async def main():
    # Training downloads ~10y per ticker, finds the setups and validates them, so keep it off the
    # event loop. (Cached after the first run; retrains if tickers, features or strategy change.)
    await asyncio.to_thread(train.xgboost, cfg.general.tickers)
    # Learn the news weight from past logged calls whose stop/target outcomes have since played out.
    params = await asyncio.to_thread(decision.refit_news)
    logging.info(f"🧮 news weight: {params['news_gamma']:.3f} log-odds per +1.0 "
          "({params['source']}, n={params['n']}, sd={params['news_gamma_sd']:.3f})")
    ticker_agent = TickerAgent("gemini-3.8-flash", 0.0)

    consumers = [
        asyncio.create_task(consumer(i, ticker_agent))
        for i in range(MAX_CONCURRENT)
    ]
    producer = asyncio.create_task(news_source())
    scanner = asyncio.create_task(daily_scan())
    for t in [producer, scanner, *consumers]:
        t.add_done_callback(_report_task_death)

    await asyncio.gather(producer, scanner, *consumers, return_exceptions=True)


if __name__ == "__main__":    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info(f"\nAsync agent shut down gracefully.")