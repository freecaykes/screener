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

from datetime import datetime, time
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


_NY = ZoneInfo("America/New_York")
_MARKET_DONE = time(16, 5)  # conservative: early-close days are treated as open until 16:05


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