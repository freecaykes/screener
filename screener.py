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
#   2. first asks the LLM whether the news can plausibly move this stock; unrelated news stops here
#   3. estimates the chance the trade reaches its profit target before its stop (train/setup_model.py)
#   4. has the LLM read the headlines IN THE CONTEXT of that setup (noise vs lasting damage / real catalyst)
#   5. prints BUY / SELL with an entry reference, stop, target and time stop -- or HOLD with the reason
#
# Triggers: a once-a-day scan after the close, new headlines, and once at startup.
# =============================================================================
import asyncio
import math
import time
from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import logging
from notifications import notify

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated*",
    module=r"yfinance",
)

import yfinance as yf


from train import features, setups, train
from agent import decision, newsfeed,TickerAgent
from config import get_config

cfg = get_config()

# =============================================================================
# CONCURRENT TICKER PROCESSOR (with semaphore)
# =============================================================================
QUEUE: asyncio.Queue = asyncio.Queue()
LAST_HEADLINES: dict[str, str] = {}
QUEUED: set[str] = set()
IN_FLIGHT: set[str] = set()
LAST_NOTIFIED: dict[str, str] = {}

_SCAN_TIME = dtime(16, 10)     # New York time: daily scan, just after the close (bars are complete by 16:05)
_NY = ZoneInfo("America/New_York")


def seconds_until_next_scan(now: Optional[datetime] = None) -> float:
    """Seconds until the next weekday SCAN_TIME (New York). Holidays still scan; they just see no new bar."""
    now = (now or datetime.now(_NY)).astimezone(_NY)
    target = datetime.combine(now.date(), _SCAN_TIME, tzinfo=_NY)
    if now >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def queue_ticker(ticker: str, trigger: str, force: bool = False) -> bool:
    """Enqueue a ticker unless a run is already queued or in flight."""
    if not force and (ticker in QUEUED or ticker in IN_FLIGHT):
        logging.info(f"[{ticker}] already queued/in flight; skipping {trigger} trigger")
        return False
    QUEUED.add(ticker)
    await QUEUE.put((ticker, trigger))
    return True


def print_result(ticker: str, result: dict) -> None:
    ind = result.get("indicators") or {}
    d = result.get("signal_detail") or {}
    provisional = bool(d.get("provisional", False))
    as_of_label = result.get("as_of_ts") or result.get("as_of", "n/a")
    logging.info(
        f"{ticker} @ {time.strftime('%H:%M:%S')} "
        f"({'PROVISIONAL intraday' if provisional else 'completed session'}: {as_of_label}), "
        f"vs {result.get('benchmark', 'benchmark')}"
    )
    reaction = result.get("price_reaction") or {}
    if reaction.get("live"):
        logging.info(
            f"   Intraday : last {reaction['last']:.2f} ({reaction['session_move_pct']:+.2f}%), "
            f"gap {reaction['gap_pct']:+.2f}%, range {reaction['range_pos']:.0%}, "
            f"volume {reaction['volume_pace']:.2f}x pace, "
            f"{reaction['session_fraction']:.0%} elapsed"
        )
    logging.info(f"   Headline : {' | '.join(result.get('headlines') or ['(none)'])}")
    dte = ind.get("days_to_earn")
    if dte is not None and math.isfinite(dte):
        logging.info(f"   Earnings : next in {dte:.0f} day(s)")

    setup = result.get("setup") or ""
    if not setup:
        reason = d.get("reason", "")
        if reason:
            logging.info(f"   Status   : HOLD -- {reason}")
        else:
            logging.info(f"   Setup    : none active -> HOLD")
        return

    match, hard = result.get("setup_match", 0.0), result.get("setup_hard", False)
    bucket = d.get("bucket", "hard" if hard else "weak")

    if setup in ("breakout_buy", "breakdown_sell"):
        what = (f"{ind.get('bo_atr', float('nan')):+.1f} ATR through the 20-day {'high' if setup == 'breakout_buy' else 'low'}, "
                f"{ind.get('rel_volume', float('nan')):.1f}x volume")
    else:
        what = (f"{'dip' if setup == 'pullback_buy' else 'rally'} of {ind.get('depth_atr', float('nan')):.1f} ATR from the 10-day "
                f"{'high' if setup == 'pullback_buy' else 'low'}")

    if hard:
        logging.info(f"   Setup    : {setup.upper()} -- FULLY FORMED ({what}; ATR {ind.get('atr_pct', float('nan')):.2f}% of price)")
    else:
        logging.info(f"   Setup    : closest is {setup.upper()}, {int(match * 100)}% matched [{bucket.upper()}] ({what}; ATR {ind.get('atr_pct', float('nan')):.2f}% of price)")
        for g in result.get("setup_gaps") or []:
           logging.info(f"              - missing: {g}")

    news = d.get("news") or result.get("news_analysis")
    if news:
        logging.info(f"   News     : sentiment {news['sentiment']:+.2f}, thesis '{news['thesis']}', materiality {news['materiality']:.2f} -- {news.get('reason', '')}")
        if news.get("plan"):
            logging.info(f"   LLM plan : {news['plan']}")
    if "p_final" in d:
        logging.info(f"   Odds     : any-day {d['p_any']:.0%} -> this setup -> final {d['p_final']:.0%} (setup evidence {d['setup_trust']:.2f}, model {d['model_trust']:.2f}, news weight {d['gamma']:.2f} [{d['gamma_source']}])")
        logging.info(f"   Value    : EV {d['ev_R']:+.2f}R per trade after {d['cost_R']:.2f}R costs ({'validated' if d['validated'] else 'NOT validated out-of-sample'})")

    plan = d.get("plan")
    if plan and (d.get("watch") or result["signal"] != "HOLD"):   # a plan for a weak match would be noise
        verb = "Buy" if d.get("direction", 1) > 0 else "Sell/short"
        logging.info(f"   Plan     : {verb} near {plan['entry_ref']:.2f} | stop {plan['stop']:.2f} ({plan['stop_pct']:.1f}%) | target {plan['target']:.2f} ({plan['target_pct']:.1f}%) | exit by {plan['max_hold_date']}")

    prefix = "PROVISIONAL " if provisional else ""
    if result["signal"] == "HOLD":
        tag = "WATCH" if d.get("watch") else "HOLD"
        logging.info(f"   Signal   : **{prefix}HOLD** [{tag}] {d.get('reason', '')}")
    else:
        logging.info(
            f"   Signal   : **{prefix}{result['signal']}** "
            f"(P(target before stop) {result['signal_confidence']:.0%})"
        )


async def consumer(id: int, ticker_agent: TickerAgent):
    while True:
        item = await QUEUE.get()
        ticker, trigger = item if isinstance(item, tuple) else (item, "news")
        QUEUED.discard(ticker)
        IN_FLIGHT.add(ticker)
        try:
            logging.info(f"Worker {id} started processing {ticker} ({trigger})")
            result = await ticker_agent.run(ticker, trigger=trigger)
            print_result(ticker, result)
            signal = result["signal"]
            detail = result.get("signal_detail") or {}
            provisional = bool(detail.get("provisional", False))
            if "HOLD" not in signal:
                previous = LAST_NOTIFIED.get(ticker)
                if previous != signal or not provisional:
                    tag = "PROVISIONAL " if provisional else ""
                    plan = detail.get("plan") or {}
                    plan_text = ""
                    if plan:
                        plan_text = (
                            f"\n Entry {plan.get('entry_ref')} | stop {plan.get('stop')} "
                            f"| target {plan.get('target')}"
                        )
                    notify(
                        f"[{ticker}] - {tag}{signal}",
                        f"Signal: **{tag}{result['signal']}** "
                        f"(P(target before stop) {result['signal_confidence']:.0%})\n"
                        f"Headline : {' | '.join(result.get('headlines') or ['(none)'])}"
                        f"{plan_text}",
                        priority="max",
                    )
                    LAST_NOTIFIED[ticker] = signal
            elif LAST_NOTIFIED.get(ticker) not in (None, "HOLD"):
                LAST_NOTIFIED[ticker] = "HOLD"

        except Exception as e:
            logging.info(f"Error processing {ticker}: {e}")
        finally:
            IN_FLIGHT.discard(ticker)
            QUEUE.task_done()


async def latest_headline(ticker: str, news_limit: int) -> Optional[str]:
    """Newest headline for `ticker`, or None. Every failure path says why it returned None."""
    try:
        # NOTE: on timeout the worker thread keeps running in the background (threads
        # can't be killed), but the loop moves on instead of waiting forever.
        items = await asyncio.wait_for(
            asyncio.to_thread(newsfeed.fetch_news, ticker, news_limit), timeout=cfg.runtime.news_fetch_timeout_sec
        )
    except asyncio.TimeoutError:
        logging.info(f"[{ticker}] news fetch timed out after {cfg.runtime.news_fetch_timeout_sec}s")
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


async def queue_new_headlines(news_limit: int) -> None:
    """Poll each ticker and enqueue it only when its newest headline changed."""
    for ticker in cfg.general.tickers:
        headline = await latest_headline(ticker, news_limit)
        if not headline:
            logging.info(f"[{ticker}] no headline available; not queueing")
            continue
        if headline == LAST_HEADLINES.get(ticker):
            logging.info(f"[{ticker}] no new headline (latest: {headline[:60]!r})")
            continue

        logging.info(
            f"[NEW NEWS DETECTED] {ticker} → queued for processing headline: {headline[:50]} ..."
        )
        if await queue_ticker(ticker, "news"):
            LAST_HEADLINES[ticker] = headline


async def news_source():
    logging.info(f"News director started — checking every {cfg.runtime.news_poll_interval_sec}s")
    # First pass pulls `news_limit` headlines per ticker from config; after that only the newest
    # headline matters for change detection, so each poll fetches a single item.
    news_limit = cfg.runtime.news_limit

    while True:
        await queue_new_headlines(news_limit)
        news_limit = 1
        await asyncio.sleep(cfg.runtime.news_poll_interval_sec)


async def daily_scan():
    """Queue every ticker once a day just after the close, when the day's bar is complete."""

    while True:
        wait = seconds_until_next_scan()
        logging.info(f"next daily scan in {wait/3600:.1f}h")
        await asyncio.sleep(wait)
        for ticker in cfg.general.tickers:
            logging.info(f"[daily scan] queueing {ticker}")
            await queue_ticker(ticker, "scan", force=True)
        await asyncio.sleep(60)  # don't fire twice for the same minute


async def price_source():
    """Event-driven quote cache: queue a ticker only when price or volume crosses a threshold."""
    if not cfg.runtime.intraday_enabled:
        logging.info("Intraday price source disabled")
        return
    logging.info(
        f"Price director started — caching quotes every "
        f"{cfg.runtime.quote_poll_interval_sec}s during the regular session"
    )
    last_price: dict[str, float] = {}
    last_pace: dict[str, float] = {}
    last_queue: dict[str, float] = {}
    session_date = ""

    while True:
        await asyncio.sleep(cfg.runtime.quote_poll_interval_sec)
        session = features.market_session(
            start=cfg.runtime.session_start, end=cfg.runtime.session_end
        )
        if not session["is_open"]:
            continue
        if session["session_date"] != session_date:
            session_date = session["session_date"]
            last_price.clear()
            last_pace.clear()
            last_queue.clear()
            LAST_NOTIFIED.clear()  # a new session may produce the same provisional label again

        try:
            snapshots = await asyncio.to_thread(
                features.fetch_quote_snapshots,
                cfg.general.tickers,
                cfg.runtime.intraday_interval,
                cfg.runtime.volume_profile_days,
                None,
                cfg.runtime.session_start,
                cfg.runtime.session_end,
            )
        except Exception as e:
            logging.info(f"quote snapshot batch failed: {type(e).__name__}: {e}")
            continue

        now = time.time()
        for ticker in cfg.general.tickers:
            snapshot = snapshots.get(ticker) or {}
            if not snapshot.get("live"):
                continue

            price = float(snapshot["last"])
            atr_pct = 0.0
            try:
                daily = await asyncio.to_thread(
                    features.fetch_history_cached, ticker, features.LIVE_PERIOD
                )
                daily = features.drop_incomplete_last_bar(daily)
                if len(daily) > 20:
                    atr = setups.atr(daily)
                    last_atr = float(atr.iloc[-1])
                    last_close = float(daily["Close"].iloc[-1])
                    if last_close > 0:
                        atr_pct = last_atr / last_close * 100.0
            except Exception as e:
                logging.debug(f"[{ticker}] ATR for trigger failed: {type(e).__name__}: {e}")

            base_price = last_price.get(ticker) or snapshot.get("prev_close") or price
            move_pct = abs(price / base_price - 1.0) * 100.0 if base_price else 0.0
            threshold_pct = max(
                cfg.runtime.price_trigger_pct, cfg.runtime.price_trigger_atr * atr_pct
            )
            volume_pace = float(snapshot.get("volume_pace") or 0.0)
            volume_crossed = (
                volume_pace >= cfg.runtime.volume_trigger_rel
                and last_pace.get(ticker, 0.0) < cfg.runtime.volume_trigger_rel
            )
            cooled = now - last_queue.get(ticker, 0.0) >= cfg.runtime.intraday_cooldown_sec

            if cooled and (volume_crossed or move_pct >= threshold_pct):
                if await queue_ticker(ticker, "price"):
                    logging.info(
                        f"[PRICE TRIGGER] {ticker} {price:.2f} "
                        f"({move_pct:+.2f}% vs last trigger, threshold {threshold_pct:.2f}%), "
                        f"volume pace {volume_pace:.2f}x"
                    )
                    last_price[ticker] = price
                    last_queue[ticker] = now

            last_pace[ticker] = volume_pace


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
    logging.info(f"news weight: {params['news_gamma']:.3f} log-odds per +1.0 ({params['source']}, n={params['n']}, sd={params['news_gamma_sd']:.3f})")
    ticker_agent = TickerAgent(cfg.llm.model, cfg.llm.temperature)

    consumers = [
        asyncio.create_task(consumer(i, ticker_agent))
        for i in range(cfg.runtime.max_concurrent)
    ]
    producer = asyncio.create_task(news_source())
    scanner = asyncio.create_task(daily_scan())
    pricer = asyncio.create_task(price_source())
    for t in [producer, scanner, pricer, *consumers]:
        t.add_done_callback(_report_task_death)

    await asyncio.gather(producer, scanner, pricer, *consumers, return_exceptions=True)


def run() -> None:
    """Blocking entry point shared by `python __main__.py` and the `stock-agent` script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nAsync agent shut down gracefully.")


if __name__ == "__main__":
    run()
