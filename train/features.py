"""
train/features.py -- the ONE place features are defined.

Training (train.py) and live inference (agent.py) both call build_features(),
so the model can never see a differently-defined feature in production than
the one it was trained on.

Design rules
------------
* Every feature is scale-free (ratios / percentages), so a model trained on
  many tickers is not just learning "which stock is this" from price levels.
* Only data that exists historically -- no news sentiment (yfinance has no
  history for it, so it cannot be trained on; it lives in the signal formula).
* Pure pandas (no pandas_ta) so there is one implementation and no library
  version drift between train and serve.
* Same data source for both sides: yf.Ticker(...).history(), dates normalised
  to tz-naive so stock and VIX rows align identically everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

FEATURE_COLS: list[str] = [
    "pullback_buy_setup",  # uptrend (Close > EMA50) AND within +-2% of EMA21
    "RSI_14",
    "price_to_ema21",
    "vix_current",
    "macd_pct",            # MACD line as % of price  (raw MACD is in price units)
    "macd_signal_pct",     # MACD signal as % of price
    "vol_20d",             # 20d std of daily returns, in %  (lets pooled trees scale by ticker vol)
]

# History needed at inference so every indicator is fully warmed up.
# EMA50 needs 50 trading days; the old period="60d" was ~41 trading days.
LIVE_PERIOD = "1y"


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


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(prices: pd.DataFrame, vix: pd.Series | None) -> pd.DataFrame:
    """Return a frame indexed by date with ['Close'] + FEATURE_COLS.

    Rows where any feature is not yet defined (indicator warm-up, missing VIX)
    are dropped, so `.iloc[-1]` is always a complete, usable feature row.
    """
    cols = ["Close"] + FEATURE_COLS
    if prices is None or prices.empty:
        return pd.DataFrame(columns=cols, dtype=float)

    close = prices["Close"].astype(float)
    out = pd.DataFrame({"Close": close})

    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()

    out["RSI_14"] = _rsi(close, 14)
    out["price_to_ema21"] = close / ema21
    out["macd_pct"] = macd / close * 100.0
    out["macd_signal_pct"] = macd_signal / close * 100.0
    out["vol_20d"] = close.pct_change().rolling(20).std() * 100.0

    setup = (close > ema50) & (out["price_to_ema21"] > 0.98) & (out["price_to_ema21"] < 1.02)
    out["pullback_buy_setup"] = setup.astype(float).where(ema50.notna())  # NaN during EMA50 warm-up

    if vix is not None and len(vix):
        out["vix_current"] = _tidy(vix).reindex(out.index, method="ffill")
    else:
        out["vix_current"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[cols].dropna(subset=FEATURE_COLS)