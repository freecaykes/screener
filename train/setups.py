"""
train/setups.py -- swing setups (pullback / breakout, long and short) and their outcomes.

Training and live inference both call build_setup_table(), so a setup means exactly the
same thing in production as in the data the model learned from.

The four setups (all evaluated on COMPLETED daily bars)
  pullback_buy  : uptrend, price has dipped >= PULLBACK_MIN_ATR below its recent high, sits at/below
                  the 21-day EMA, and shows a REVERSAL candle (down yesterday, green today, closing in
                  the upper half of its range). This is the practical stand-in for "close to a local
                  minimum": a minimum is only knowable afterwards, so we wait for the first sign the
                  dip has stopped, and fire only on that first bar.
  breakout_buy  : close above the prior BREAKOUT_LOOKBACK-day high on >= 1.3x volume, above the 50 EMA.
  rally_sell    : mirror of pullback_buy in a downtrend (rally into resistance that has stalled).
  breakdown_sell: mirror of breakout_buy (close below the prior 20-day low on volume).

Outcome (the "triple barrier"): after a setup fires on bar t, enter at the NEXT open and ask whether
price reaches the profit target (TP_ATR x ATR) before the stop (SL_ATR x ATR) within MAX_HOLD bars.
A tight stop that survives means the entry was close to the low (or high, for sells) -- which is
what "as close to a local minimum as possible" means in measurable terms.

All thresholds below are strategy policy, not fitted values.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from train import features

# ---- strategy policy ---------------------------------------------------------
TP_ATR = 2.0            # profit target, in ATRs from entry
SL_ATR = 1.0            # stop, in ATRs from entry  (so the trade is 2R if it works)
MAX_HOLD = 10           # time stop, in trading bars after entry
COST_PCT = 0.10         # assumed round-trip cost + slippage, % of price (a short may cost more)
PULLBACK_MIN_ATR = 1.5  # dip from the 10-bar high must be at least this many ATRs
PULLBACK_WINDOW = 10
BREAKOUT_LOOKBACK = 20
BREAKOUT_MIN_REL_VOLUME = 1.3
EMA_TOUCH_ATR = 0.25    # "at or below the 21 EMA" tolerance, in ATRs

SETUPS = ["pullback_buy", "breakout_buy", "rally_sell", "breakdown_sell"]

# Direction-normalised features: for sells the return-like ones are multiplied by -1 so that
# "positive" always means "in the direction of the trade". That lets buys and sells share one model.
SETUP_FEATURES: list[str] = [
    "direction",           # +1 buy setups, -1 sell setups
    "is_breakout",         # 1 breakout/breakdown, 0 pullback/rally
    "depth_atr",           # buys: dip from 10-bar high; sells: rally from 10-bar low (in ATRs)
    "bo_atr",              # how far through the 20-bar high (buys) / low (sells), in ATRs
    "ema21_gap_atr_s",     # distance beyond the 21 EMA in the trade's favour (pullbacks: below it for buys)
    "trend_slope_atr_s",   # 50 EMA 10-bar slope in ATRs, signed by direction
    "atr_pct",             # ATR as % of price
    "rel_volume",
    "gap_pct_s",
    "ret_5d_s",
    "rs_5d_s",             # 5d return relative to the benchmark, signed
    "rs_20d_s",
    "mom_12_1_s",
    "range_pos_s",         # position in the 52-week range, signed (buys: high = near highs)
    "vix_current",
    "days_to_earn",        # optional (NaN when unknown)
]
OPTIONAL_SETUP_FEATURES = ["days_to_earn"]
REQUIRED_SETUP_FEATURES = [c for c in SETUP_FEATURES if c not in OPTIONAL_SETUP_FEATURES]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def atr(prices: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = prices["High"], prices["Low"], prices["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build_setup_table(
    prices: pd.DataFrame,
    vix: Optional[pd.Series],
    bench: Optional[pd.Series],
    earnings: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """One row per usable date: Close, setup, direction, atr_pct + SETUP_FEATURES.

    `setup` is "" when none of the four setups is active. Rows with any undefined REQUIRED
    feature (indicator warm-up etc.) are dropped, so `.iloc[-1]` is always a complete row.
    """
    cols = ["Close", "setup"] + SETUP_FEATURES
    feats = features.build_features(prices, vix, bench, earnings)  # generic, shared feature set
    if feats.empty:
        return pd.DataFrame(columns=cols)

    close, high, low = prices["Close"].astype(float), prices["High"].astype(float), prices["Low"].astype(float)
    a = atr(prices)
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()

    up_trend = (close > ema50) & (ema50 > ema50.shift(10))
    down_trend = (close < ema50) & (ema50 < ema50.shift(10))

    hi_w = close.rolling(PULLBACK_WINDOW).max()
    lo_w = close.rolling(PULLBACK_WINDOW).min()
    down_depth = (hi_w - close) / a             # how far we have dipped from the recent high
    up_height = (close - lo_w) / a              # how far we have rallied from the recent low
    prior_hi = high.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    prior_lo = low.rolling(BREAKOUT_LOOKBACK).min().shift(1)
    bo_up = (close - prior_hi) / a
    bo_dn = (prior_lo - close) / a
    ema21_gap = (ema21 - close) / a             # > 0 when below the 21 EMA

    # Reversal candle = "the move has stopped": yesterday went the other way, and today is a candle in
    # the trade's direction that closes in the favourable half of its range. Hammers that undercut
    # yesterday's low and then close up count (they are often the best bottoming candles), and
    # requiring the previous bar to be against us means only the FIRST reversal bar fires, not every
    # up-day of the recovery.
    open_ = prices["Open"].astype(float)
    close_pos = (close - low) / (high - low).replace(0, np.nan)   # 0 = closed at the low, 1 = at the high
    stab_up = (close.shift(1) < close.shift(2)) & (close > open_) & (close_pos >= 0.5)
    stab_dn = (close.shift(1) > close.shift(2)) & (close < open_) & (close_pos <= 0.5)

    rel_vol = feats["rel_volume"].reindex(close.index)
    pullback_buy = up_trend & (down_depth >= PULLBACK_MIN_ATR) & (ema21_gap >= -EMA_TOUCH_ATR) & stab_up
    breakout_buy = (bo_up > 0) & (rel_vol >= BREAKOUT_MIN_REL_VOLUME) & (close > ema50)
    rally_sell = down_trend & (up_height >= PULLBACK_MIN_ATR) & (ema21_gap <= EMA_TOUCH_ATR) & stab_dn
    breakdown_sell = (bo_dn > 0) & (rel_vol >= BREAKOUT_MIN_REL_VOLUME) & (close < ema50)

    setup = np.select(
        [pullback_buy, breakout_buy, rally_sell, breakdown_sell],
        SETUPS, default="",
    )
    d = np.select([setup == "pullback_buy", setup == "breakout_buy", setup == "rally_sell", setup == "breakdown_sell"],
                  [1.0, 1.0, -1.0, -1.0], default=0.0)
    ds = pd.Series(d, index=close.index)
    buy = ds > 0

    hi252 = close.rolling(252, min_periods=252).max()
    lo252 = close.rolling(252, min_periods=252).min()
    range_pos = (close - lo252) / (hi252 - lo252)

    t = pd.DataFrame(index=close.index)
    t["Close"] = close
    t["setup"] = setup
    t["direction"] = ds
    t["is_breakout"] = np.isin(setup, ["breakout_buy", "breakdown_sell"]).astype(float)
    t["depth_atr"] = np.where(buy, down_depth, up_height)
    t["bo_atr"] = np.where(buy, bo_up, bo_dn)
    t["ema21_gap_atr_s"] = ema21_gap * ds
    t["trend_slope_atr_s"] = ((ema50 - ema50.shift(10)) / a) * ds
    t["atr_pct"] = a / close * 100.0
    t["rel_volume"] = rel_vol
    t["gap_pct_s"] = feats["gap_pct"].reindex(close.index) * ds
    t["ret_5d_s"] = feats["ret_5d"].reindex(close.index) * ds
    t["rs_5d_s"] = feats["rs_5d"].reindex(close.index) * ds
    t["rs_20d_s"] = feats["rs_20d"].reindex(close.index) * ds
    t["mom_12_1_s"] = feats["mom_12_1"].reindex(close.index) * ds
    t["range_pos_s"] = np.where(buy, range_pos, 1.0 - range_pos)
    t["vix_current"] = feats["vix_current"].reindex(close.index)
    t["days_to_earn"] = feats["days_to_earn"].reindex(close.index)

    t = t.reindex(feats.index).replace([np.inf, -np.inf], np.nan)
    return t[cols].dropna(subset=["Close"] + REQUIRED_SETUP_FEATURES)


# ---------------------------------------------------------------------------
# Outcomes: the triple barrier
# ---------------------------------------------------------------------------
def barrier_outcome(o, h, l, c, i: int, d: float, atr_pct: float,
                    tp_atr: float = TP_ATR, sl_atr: float = SL_ATR, hold: int = MAX_HOLD):
    """Outcome of a trade signalled on completed bar `i`, entered at the NEXT open.

    o, h, l, c : arrays of open/high/low/close
    d          : +1 long, -1 short
    Returns (y, R, bars) or None if the window isn't complete yet:
      y = 1 if the profit target is hit before the stop, else 0 (stop or time-out)
      R = result in units of the stop distance (target hit = tp_atr/sl_atr, stop = -1,
          time-out = marked to the last close)
      bars = bars held

    If one bar touches BOTH barriers we assume the stop came first (the conservative choice).
    A stop that gaps through is filled at the OPEN (so R can be worse than -1); targets are
    credited only at the target price (favourable gaps are not credited).
    """
    n = len(c)
    if i + 1 >= n:
        return None
    entry = o[i + 1]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr_pct) or atr_pct <= 0:
        return None
    tp_dist, sl_dist = tp_atr * atr_pct / 100.0, sl_atr * atr_pct / 100.0
    target, stop = entry * (1.0 + d * tp_dist), entry * (1.0 - d * sl_dist)
    last = i + hold
    for k in range(i + 1, min(last, n - 1) + 1):
        if d > 0:
            hit_stop, hit_target = l[k] <= stop, h[k] >= target
        else:
            hit_stop, hit_target = h[k] >= stop, l[k] <= target
        if hit_stop:  # includes the both-in-one-bar case
            # a gap through the stop fills at the open (worse than the stop); otherwise at the stop
            fill = min(stop, o[k]) if d > 0 else max(stop, o[k])
            return 0, float(d * (fill / entry - 1.0) / sl_dist), k - i
        if hit_target:
            return 1, tp_atr / sl_atr, k - i
    if last > n - 1:
        return None  # window not complete and nothing hit yet
    return 0, float(d * (c[last] / entry - 1.0) / sl_dist), hold


def label_rows(prices: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add y, R, bars to the rows of `table` (each row's outcome from its own date). Unmatured rows dropped."""
    o, h, l, c = (prices[k].to_numpy(float) for k in ["Open", "High", "Low", "Close"])
    pos = prices.index.get_indexer(table.index)
    ys, rs, bs, keep = [], [], [], []
    for (date, row), i in zip(table.iterrows(), pos):
        d = row["direction"] if row["direction"] != 0 else 1.0
        res = barrier_outcome(o, h, l, c, int(i), float(d), float(row["atr_pct"])) if i >= 0 else None
        if res is None:
            continue
        keep.append(date); ys.append(res[0]); rs.append(res[1]); bs.append(res[2])
    out = table.loc[keep].copy()
    out["y"], out["R"], out["bars"] = ys, rs, bs
    return out


def add_trading_days(start, n: int) -> pd.Timestamp:
    """The date n trading days after `start` (start itself is day 0). NYSE calendar if available."""
    start = pd.Timestamp(start)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    start = start.normalize()
    try:
        import pandas_market_calendars as mcal
        days = mcal.get_calendar("NYSE").valid_days(
            start_date=start + pd.Timedelta(days=1), end_date=start + pd.Timedelta(days=2 * n + 10))
        return days[n - 1].tz_localize(None).normalize()
    except ImportError:
        return start + pd.offsets.BDay(n)