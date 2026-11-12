"""
train/features.py -- the ONE place features are defined.

Training (train.py) and live inference (agent.py) both call build_features(), so the
model can never see a differently-defined feature in production than the one it was
trained on.

Design rules
------------
* Scale-free features only (ratios / percentages), so a model is not just learning
  "which stock is this" from price levels.
* Only data that exists historically -- no news sentiment (yfinance has no history for
  it, so it cannot be trained on; it lives in the decision layer instead).
* Market-relative: the model predicts the stock's move MINUS a benchmark's move
  (default SPY). Market direction is most of a stock's 3-day variance and is not what
  these features can predict, so it is removed from the target and the features
  describe the stock relative to the market.
* Pure pandas (no pandas_ta): one implementation, no library-version drift.
* Same data source for both sides: yf.Ticker(...).history(), dates normalised to
  tz-naive so stock, benchmark and VIX rows align identically everywhere.

Feature groups (each has published research behind it, unlike generic oscillators)
  short-term reversal : ret_1d, ret_5d, rs_5d (5d return relative to the market)
  relative strength   : rs_20d
  medium-term momentum: mom_12_1 (12-month return skipping the latest month)
  52-week range       : dist_52w_high
  participation       : rel_volume, gap_pct
  market sensitivity  : beta_60d
  risk / regime       : vol_20d, excess_vol_20d, vix_current
  pullback thesis     : price_to_ema21, pullback_buy_setup
  earnings (optional) : days_to_earn, days_since_earn  (NaN when unavailable)
"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated*",
    module=r"yfinance",
)

import logging
import yfinance as yf

# ---------------------------------------------------------------------------
# Benchmark (what "relative" means)
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARK = "SPY"
# Per-ticker override, e.g. {"NOW": "XLK"} to measure against a sector ETF instead.
BENCHMARK_OVERRIDES: dict[str, str] = {}


def benchmark_for(ticker: str) -> str:
    return BENCHMARK_OVERRIDES.get(ticker, DEFAULT_BENCHMARK)


# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "pullback_buy_setup",  # uptrend (Close > EMA50) AND within +-2% of EMA21
    "price_to_ema21",
    "vix_current",
    "vol_20d",             # 20d std of the stock's daily returns, %
    "excess_vol_20d",      # 20d std of (stock - benchmark) daily returns, %  <- sizes the uncertainty
    "ret_1d",              # last day's return, %
    "ret_5d",              # last 5 days' return, %
    "rs_5d",               # ret_5d minus the benchmark's, %
    "rs_20d",              # 20d return minus the benchmark's, %
    "mom_12_1",            # return from t-252 to t-21 (skips the most recent month), %
    "dist_52w_high",       # % below the trailing 252-day high (<= 0)
    "rel_volume",          # volume / 20d average volume
    "gap_pct",             # open vs previous close, %
    "beta_60d",            # 60d beta to the benchmark
    "days_to_earn",        # calendar days to the next earnings date (NaN if unknown)
    "days_since_earn",     # calendar days since the last earnings date (NaN if unknown)
]
# Earnings dates come from a flaky endpoint, so those two may be NaN; XGBoost handles
# missing values natively. Every other feature is required for a row to be usable.
OPTIONAL_COLS: list[str] = ["days_to_earn", "days_since_earn"]
REQUIRED_COLS: list[str] = [c for c in FEATURE_COLS if c not in OPTIONAL_COLS]

# mom_12_1 and dist_52w_high need ~253 trading days of warm-up, so fetch 2 years live.
LIVE_PERIOD = "2y"

_EARNINGS_MAX_GAP_DAYS = 120  # quarters are ~91 days apart; a bigger gap means missing data

_NY = ZoneInfo("America/New_York")
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)

_DAILY_CACHE: dict[tuple[str, str], tuple[datetime, pd.DataFrame]] = {}
_EARNINGS_CACHE: dict[str, tuple[object, pd.DatetimeIndex]] = {}
_INTRADAY_CACHE: dict[tuple[str, str, str], tuple[datetime, pd.DataFrame]] = {}
_VOLUME_PROFILE_CACHE: dict[tuple[str, int, str], tuple[object, pd.Series]] = {}


# ---------------------------------------------------------------------------
# Data access (identical for training and inference)
# ---------------------------------------------------------------------------
def _naive_dates(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # drop tz, keep the exchange-local date
    return idx.normalize()


def _tidy(obj):
    obj = obj.copy()
    obj.index = _naive_dates(obj.index)
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    return obj


def align_series(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Put `series` (e.g. benchmark closes) on `index`, forward-filling gaps."""
    return _tidy(series).reindex(index, method="ffill")


def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    """Daily OHLCV, split/dividend adjusted, tz-naive date index. Empty frame on failure."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return _tidy(df)


def _ny_now(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(_NY)
    return now.replace(tzinfo=_NY) if now.tzinfo is None else now.astimezone(_NY)


def fetch_history_cached(
    ticker: str,
    period: str = LIVE_PERIOD,
    now: Optional[datetime] = None,
    max_age_sec: int = 900,
    force: bool = False,
) -> pd.DataFrame:
    """Daily history reused across intraday polls; `force` refreshes after the close."""
    current = _ny_now(now)
    key = (ticker, period)
    cached = _DAILY_CACHE.get(key)
    if cached and not force and (current - cached[0]).total_seconds() <= max_age_sec:
        return cached[1].copy()
    df = fetch_history(ticker, period)
    _DAILY_CACHE[key] = (current, df)
    return df.copy()


def _tidy_intraday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(_NY).tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_intraday(
    ticker: str,
    interval: str = "1m",
    period: str = "1d",
    now: Optional[datetime] = None,
    max_age_sec: int = 20,
) -> pd.DataFrame:
    """Intraday OHLCV bars, timezone-naive New York time. Empty frame on failure."""
    current = _ny_now(now)
    key = (ticker, interval, period)
    cached = _INTRADAY_CACHE.get(key)
    if cached and (current - cached[0]).total_seconds() <= max_age_sec:
        return cached[1].copy()
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True, prepost=False
        )
    except Exception as e:
        logging.info(f"  {ticker}: intraday fetch failed ({type(e).__name__}: {e})")
        df = pd.DataFrame()
    if df is None or df.empty:
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    else:
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        df = _tidy_intraday(df)
    _INTRADAY_CACHE[key] = (current, df)
    return df.copy()


def fetch_vix(period: str) -> pd.Series:
    df = fetch_history("^VIX", period)
    return df["Close"] if not df.empty else pd.Series(dtype=float)


def fetch_bench(ticker: str, period: str) -> pd.Series:
    """Close series of the benchmark used for `ticker`."""
    df = fetch_history(benchmark_for(ticker), period)
    return df["Close"] if not df.empty else pd.Series(dtype=float)


_earnings_warned: set[str] = set()


def fetch_earnings_dates(ticker: str, limit: int = 60) -> pd.DatetimeIndex:
    """Known earnings dates (past and scheduled), tz-naive. Empty index on any failure.

    NOTE: depends on a Yahoo endpoint that is sometimes unavailable or rate limited; the
    model and decision layer both work without it (features become NaN).
    """
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        if df is None or len(df) == 0:
            raise ValueError("no earnings dates returned")
        return pd.DatetimeIndex(sorted(set(_naive_dates(df.index))))
    except Exception as e:
        if ticker not in _earnings_warned:  # warn once per ticker, not on every run
            _earnings_warned.add(ticker)
            logging.info(f"  {ticker}: earnings dates unavailable ({type(e).__name__}: {e}); "
                  "earnings features will be NaN")
        return pd.DatetimeIndex([])


def fetch_earnings_dates_cached(
    ticker: str, limit: int = 60, now: Optional[datetime] = None
) -> pd.DatetimeIndex:
    """Earnings dates cached for the current New York calendar day."""
    current = _ny_now(now)
    cached = _EARNINGS_CACHE.get(ticker)
    if cached and cached[0] == current.date():
        return cached[1].copy()
    dates = fetch_earnings_dates(ticker, limit)
    _EARNINGS_CACHE[ticker] = (current.date(), dates)
    return dates.copy()


_MARKET_DONE = time(16, 5)  # conservative: early-close days are treated as open until 16:05


def _parse_hhmm(value: Optional[str], default: time) -> time:
    if not value:
        return default
    try:
        hour, minute = (int(part) for part in value.split(":", 1))
        return time(hour, minute)
    except (AttributeError, ValueError):
        return default


def _time_minutes(value: time) -> float:
    return value.hour * 60.0 + value.minute + value.second / 60.0


def _interval_minutes(interval: str) -> int:
    value, unit = interval[:-1], interval[-1:]
    try:
        amount = int(value)
    except ValueError:
        return 5
    if unit == "h":
        return amount * 60
    if unit == "d":
        return amount * 390
    return max(amount, 1)


def market_session(
    now: Optional[datetime] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Current US regular-session state and the elapsed fraction of that session."""
    current = _ny_now(now)
    opens = _parse_hhmm(start, _SESSION_OPEN)
    closes = _parse_hhmm(end, _SESSION_CLOSE)
    trading_day = current.weekday() < 5
    current_min = _time_minutes(current.time())
    open_min, close_min = _time_minutes(opens), _time_minutes(closes)
    total = max(close_min - open_min, 1.0)
    elapsed = min(max(current_min - open_min, 0.0), total)
    return {
        "is_trading_day": trading_day,
        "is_open": bool(trading_day and open_min <= current_min < close_min),
        "session_date": current.date().isoformat(),
        "session_fraction": elapsed / total,
        "elapsed_minutes": elapsed,
        "total_minutes": total,
        "now": current.isoformat(timespec="seconds"),
    }


def _fallback_volume_fraction(now: datetime, start: time, end: time) -> float:
    open_min, close_min = _time_minutes(start), _time_minutes(end)
    total = max(close_min - open_min, 1.0)
    return min(max((_time_minutes(now.time()) - open_min) / total, 0.0), 1.0)


def fetch_volume_profile(
    ticker: str,
    days: int = 20,
    interval: str = "5m",
    now: Optional[datetime] = None,
) -> pd.Series:
    """Average cumulative volume fraction by minute of the regular session.

    Index values are minutes since midnight at the END of each intraday bucket, so the
    current expected fraction can be interpolated without guessing how far into a bar we are.
    Empty series on failure; callers then fall back to elapsed-session time.
    """
    current = _ny_now(now)
    key = (ticker, days, interval)
    cached = _VOLUME_PROFILE_CACHE.get(key)
    if cached and cached[0] == current.date():
        return cached[1].copy()

    lookback = min(max(days * 3, 10), 60)
    bars = fetch_intraday(
        ticker, interval=interval, period=f"{lookback}d", now=current, max_age_sec=3600
    )
    if bars.empty:
        profile = pd.Series(dtype=float)
    else:
        bars = bars[bars.index.date < current.date()]
        bars = bars.between_time(_SESSION_OPEN.strftime("%H:%M"), "15:59")
        bucket_minutes = _interval_minutes(interval)
        frames = []
        for _, day in bars.groupby(bars.index.date):
            total = float(day["Volume"].sum())
            if total <= 0:
                continue
            frac = day["Volume"].cumsum() / total
            minutes = [
                _time_minutes(t) + bucket_minutes for t in day.index.time
            ]
            frames.append(pd.Series(frac.to_numpy(), index=minutes))
        profile = (
            pd.concat(frames, axis=1).mean(axis=1).sort_index()
            if frames
            else pd.Series(dtype=float)
        )
        profile = profile[~profile.index.duplicated(keep="last")].clip(0.0, 1.0)
        if len(profile):
            profile.iloc[-1] = 1.0

    _VOLUME_PROFILE_CACHE[key] = (current.date(), profile)
    return profile.copy()


def expected_volume_fraction(
    profile: Optional[pd.Series],
    now: Optional[datetime] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> float:
    """Fraction of a normal day's volume expected to have traded by `now`."""
    current = _ny_now(now)
    opens = _parse_hhmm(start, _SESSION_OPEN)
    closes = _parse_hhmm(end, _SESSION_CLOSE)
    if profile is None or len(profile) == 0:
        return _fallback_volume_fraction(current, opens, closes)
    current_min = _time_minutes(current.time())
    xp = profile.index.to_numpy(float)
    fp = profile.to_numpy(float)
    return float(min(max(np.interp(current_min, xp, fp), 0.0), 1.0))


def _without_today(prices: pd.DataFrame, session_date: date) -> pd.DataFrame:
    if prices is None or prices.empty:
        return prices
    return prices[prices.index.date != session_date]


def build_live_bar(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    profile: Optional[pd.Series] = None,
    now: Optional[datetime] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Append today's live package as a synthetic daily bar plus a reaction snapshot.

    The synthetic volume is solved so that `rel_volume` in build_features equals the
    pace-adjusted volume ratio, not the incomplete cumulative volume.
    """
    current = _ny_now(now)
    session_date = current.date()
    empty_meta = {
        "bar_complete": False,
        "as_of_ts": current.isoformat(timespec="seconds"),
        "session_date": session_date.isoformat(),
        "live": False,
    }
    if daily is None or daily.empty or intraday is None or intraday.empty:
        return (daily.copy() if daily is not None else pd.DataFrame()), empty_meta

    day = intraday[intraday.index.date == session_date]
    if day.empty:
        return daily.copy(), empty_meta

    completed = _without_today(daily, session_date)
    if completed.empty:
        return daily.copy(), empty_meta

    snapshot_time = _ny_now(day.index[-1])
    last = float(day["Close"].iloc[-1])
    session_open = float(day["Open"].iloc[0])
    session_high = float(day["High"].max())
    session_low = float(day["Low"].min())
    cumulative_volume = float(day["Volume"].sum())
    prev_close = float(completed["Close"].iloc[-1])
    expected = expected_volume_fraction(profile, snapshot_time, start, end)
    expected = min(max(expected, 0.02), 1.0)
    avg_volume = float(completed["Volume"].tail(20).mean())
    if not np.isfinite(avg_volume) or avg_volume <= 0:
        avg_volume = max(cumulative_volume, 1.0)
    volume_pace = cumulative_volume / (avg_volume * expected)

    prior_19 = float(completed["Volume"].tail(19).sum())
    pace_for_volume = min(max(volume_pace, 0.0), 19.5)
    synthetic_volume = (
        pace_for_volume * prior_19 / max(20.0 - pace_for_volume, 1.0)
        if prior_19 > 0
        else avg_volume * volume_pace
    )

    row = pd.DataFrame(
        [{
            "Open": session_open,
            "High": max(session_high, last),
            "Low": min(session_low, last),
            "Close": last,
            "Volume": max(synthetic_volume, 0.0),
        }],
        index=pd.DatetimeIndex([pd.Timestamp(session_date)]),
    )
    prices = pd.concat([completed, row]).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    session_range = max(session_high - session_low, 1e-12)
    meta = {
        "bar_complete": False,
        "as_of_ts": snapshot_time.isoformat(timespec="seconds"),
        "session_date": session_date.isoformat(),
        "live": True,
        "last": last,
        "prev_close": prev_close,
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "session_volume": cumulative_volume,
        "session_move_pct": (last / prev_close - 1.0) * 100.0 if prev_close else 0.0,
        "gap_pct": (session_open / prev_close - 1.0) * 100.0 if prev_close else 0.0,
        "range_pos": (last - session_low) / session_range,
        "session_fraction": expected,
        "volume_pace": volume_pace,
        "avg_volume_20d": avg_volume,
    }
    return prices, meta


def fetch_live_prices(
    ticker: str,
    interval: str = "1m",
    profile_days: int = 20,
    now: Optional[datetime] = None,
    force_daily: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Daily history plus a synthetic live bar for `ticker`."""
    current = _ny_now(now)
    daily = fetch_history_cached(ticker, LIVE_PERIOD, now=current, force=force_daily)
    intraday = fetch_intraday(ticker, interval=interval, period="1d", now=current)
    profile = fetch_volume_profile(ticker, days=profile_days, interval="5m", now=current)
    return build_live_bar(daily, intraday, profile, now=current, start=start, end=end)


def _live_close_series(symbol: str, now: Optional[datetime] = None) -> pd.Series:
    current = _ny_now(now)
    daily = fetch_history_cached(symbol, LIVE_PERIOD, now=current)
    if daily is None or daily.empty:
        return pd.Series(dtype=float)
    intraday = fetch_intraday(symbol, interval="1m", period="1d", now=current)
    series = daily["Close"].copy()
    if intraday is not None and not intraday.empty:
        session_date = current.date()
        day = intraday[intraday.index.date == session_date]
        if not day.empty:
            series = series[series.index.date != session_date]
            series.loc[pd.Timestamp(session_date)] = float(day["Close"].iloc[-1])
    return series.sort_index()


def fetch_vix_live(now: Optional[datetime] = None) -> pd.Series:
    return _live_close_series("^VIX", now)


def fetch_bench_live(ticker: str, now: Optional[datetime] = None) -> pd.Series:
    return _live_close_series(benchmark_for(ticker), now)


def fetch_quote_snapshots(
    tickers: list[str],
    interval: str = "1m",
    profile_days: int = 20,
    now: Optional[datetime] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, dict]:
    """Live snapshot per ticker for the event-driven price loop."""
    current = _ny_now(now)
    out: dict[str, dict] = {}
    for ticker in tickers:
        try:
            _, meta = fetch_live_prices(
                ticker,
                interval=interval,
                profile_days=profile_days,
                now=current,
                start=start,
                end=end,
            )
            out[ticker] = meta
        except Exception as e:
            logging.info(f"  {ticker}: quote snapshot failed ({type(e).__name__}: {e})")
            out[ticker] = {"live": False, "as_of_ts": current.isoformat(timespec="seconds")}
    return out


def drop_incomplete_last_bar(prices: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """Drop today's bar if the US session isn't over yet.

    Training only ever sees COMPLETED sessions. During market hours yfinance returns a
    partial bar (volume so far, last price as 'close'), which would make rel_volume,
    ret_1d, gap etc. systematically different from what the model was trained on. Live
    inference therefore uses the last completed session; `as_of` then matches training.
    """
    if prices is None or prices.empty:
        return prices
    now = now or datetime.now(_NY)
    now = now.replace(tzinfo=_NY) if now.tzinfo is None else now.astimezone(_NY)
    if prices.index[-1].date() == now.date() and now.weekday() < 5 and now.time() < _MARKET_DONE:
        return prices.iloc[:-1]
    return prices


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _earnings_features(index: pd.DatetimeIndex, earnings: Optional[pd.DatetimeIndex]):
    nan = np.full(len(index), np.nan)
    if earnings is None or len(earnings) == 0:
        return nan, nan.copy()
    e = np.sort(pd.DatetimeIndex(earnings).values.astype("datetime64[ns]"))
    d = index.values.astype("datetime64[ns]")
    one_day = np.timedelta64(1, "D")

    nxt = np.searchsorted(e, d, side="left")          # first earnings on/after d
    prv = np.searchsorted(e, d, side="right") - 1     # last earnings on/before d
    to = np.where(nxt < len(e), (e[np.minimum(nxt, len(e) - 1)] - d) / one_day, np.nan)
    since = np.where(prv >= 0, (d - e[np.maximum(prv, 0)]) / one_day, np.nan)
    # a gap bigger than a quarter means the list doesn't cover this date -> unknown, not "far"
    to = np.where(to > _EARNINGS_MAX_GAP_DAYS, np.nan, to)
    since = np.where(since > _EARNINGS_MAX_GAP_DAYS, np.nan, since)
    return to.astype(float), since.astype(float)


def build_features(
    prices: pd.DataFrame,
    vix: Optional[pd.Series],
    bench: Optional[pd.Series],
    earnings: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Return a frame indexed by date with ['Close'] + FEATURE_COLS.

    Rows where any REQUIRED feature is undefined (indicator warm-up, missing VIX or
    benchmark) are dropped, so `.iloc[-1]` is always a complete, usable feature row.
    """
    cols = ["Close"] + FEATURE_COLS
    if prices is None or prices.empty or bench is None or len(bench) == 0:
        return pd.DataFrame(columns=cols, dtype=float)

    close = prices["Close"].astype(float)
    open_ = prices["Open"].astype(float)
    volume = prices["Volume"].astype(float)
    b = align_series(bench.astype(float), close.index)

    r, rb = close.pct_change(), b.pct_change()
    out = pd.DataFrame({"Close": close})

    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["price_to_ema21"] = close / ema21
    setup = (close > ema50) & (out["price_to_ema21"] > 0.98) & (out["price_to_ema21"] < 1.02)
    out["pullback_buy_setup"] = setup.astype(float).where(ema50.notna())  # NaN during warm-up

    out["vol_20d"] = r.rolling(20).std() * 100.0
    out["excess_vol_20d"] = (r - rb).rolling(20).std() * 100.0
    out["ret_1d"] = r * 100.0
    out["ret_5d"] = close.pct_change(5) * 100.0
    out["rs_5d"] = out["ret_5d"] - b.pct_change(5) * 100.0
    out["rs_20d"] = (close.pct_change(20) - b.pct_change(20)) * 100.0
    out["mom_12_1"] = (close.shift(21) / close.shift(252) - 1.0) * 100.0
    out["dist_52w_high"] = (close / close.rolling(252, min_periods=252).max() - 1.0) * 100.0
    out["rel_volume"] = volume / volume.rolling(20).mean()
    out["gap_pct"] = (open_ / close.shift(1) - 1.0) * 100.0
    out["beta_60d"] = r.rolling(60).cov(rb) / rb.rolling(60).var()

    if vix is not None and len(vix):
        out["vix_current"] = align_series(vix, out.index)
    else:
        out["vix_current"] = np.nan

    out["days_to_earn"], out["days_since_earn"] = _earnings_features(out.index, earnings)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[cols].dropna(subset=REQUIRED_COLS)
