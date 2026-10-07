"""
train/timing.py -- estimate WHEN the XGBoost delta is expected to play out,
and check whether the model has any out-of-sample skill at each horizon.

How it works
------------
One regressor per horizon h in HORIZONS (trading days ahead). Predicting all of
them for the same feature row gives an expected cumulative-move curve mu(h),
and the "when" is read off the curve:

  * reach_horizon : first horizon where the move (in the direction of the
                    headline signal) has reached `reach_frac` (80%) of its peak
  * peak_horizon  : horizon where the move is largest

Validation gate
---------------
Every horizon is scored with purged walk-forward validation (see
walk_forward_validate). Only horizons that show out-of-sample skill are allowed
to feed the timing estimate; if too few pass, the manifest says so instead of
printing a confident-looking date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:  # optional: exact NYSE holidays
    import pandas_market_calendars as mcal
except ImportError:  # pragma: no cover
    mcal = None

HORIZONS: list[int] = [1, 2, 3, 5, 8, 10, 15, 20]  # trading days ahead
BASE_HORIZON: int = 3  # what the app calls `predicted_delta`

# ---- validation gate (tune these) -----------------------------------------
# A horizon is trusted only if its out-of-sample IC is both big enough to matter and
# statistically distinguishable from zero. Fold ICs come from disjoint time blocks, so
# their spread already reflects overlapping targets and same-day cross-ticker correlation;
# a fixed IC cutoff alone does not (it passed pure noise ~11% of the time in testing).
MIN_IC = 0.02                 # mean out-of-sample Spearman IC needed to trust a horizon
MIN_T_STAT = 2.5              # ...and mean(IC) / stderr(IC across folds) must reach this
MIN_FOLDS = 5                 # ...and at least this many folds must have been scored
MIN_TRUSTED_HORIZONS = 3      # a curve shape needs >= 3 points
REQUIRE_SKILL = True          # False = use every horizon regardless of validation


# ---------------------------------------------------------------------------
# Targets / models
# ---------------------------------------------------------------------------
def target_cols() -> list[str]:
    return [f"target_delta_{h}" for h in HORIZONS]


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """target_delta_{h} = % change of Close from t to t+h trading days.

    Compute this on the FULL price series (before any row dropping) so the
    shift is by real trading days. Tail rows get NaN for longer horizons.
    """
    for h in HORIZONS:
        df[f"target_delta_{h}"] = (df["Close"].shift(-h) - df["Close"]) / df["Close"] * 100
    return df


def make_regressor() -> XGBRegressor:
    # Deliberately modest capacity: daily returns are mostly noise.
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def fit_horizon_models(D: pd.DataFrame, feature_cols: list[str], min_rows: int = 500) -> dict[int, XGBRegressor]:
    """Fit one model per horizon on the whole dataset D (features + target_delta_* columns)."""
    models: dict[int, XGBRegressor] = {}
    for h in HORIZONS:
        col = f"target_delta_{h}"
        sub = D[D[col].notna()]
        if len(sub) < min_rows:
            continue
        models[h] = make_regressor().fit(sub[feature_cols], sub[col])
    if BASE_HORIZON not in models:
        raise ValueError(f"Not enough data to train the base {BASE_HORIZON}-day model")
    return models


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def walk_forward_validate(
    D: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 8,
    first_test_frac: float = 0.4,
) -> dict[int, dict]:
    """Expanding-window, date-based, purged walk-forward validation.

    D must have a `date` column (plus features and target_delta_* columns) and
    may pool many tickers. Splitting is by DATE, never by row, so no two rows
    from the same day straddle train/test (tickers move together on market days).

    Purge: a training row at date position p has a target that looks h trading
    days ahead, so it is only used if p + h < start of the test block. That is
    what stops the overlapping multi-day targets leaking test-period prices into
    training.

    Metrics per horizon (pooled over all test rows in a fold):
      ic       Spearman rank correlation between prediction and realised move
               (one value per fold; the gate uses their mean and t-statistic)
      hit_rate share of test rows where sign(pred) == sign(realised)
      base_up  share of test rows that actually went up (the hit-rate to beat)
    """
    dates = np.sort(D["date"].unique())
    n = len(dates)
    edges = np.linspace(int(n * first_test_frac), n, n_folds + 1).astype(int)

    out: dict[int, dict] = {}
    for h in HORIZONS:
        col = f"target_delta_{h}"
        ics: list[float] = []
        hits: list[float] = []
        ups: list[float] = []
        n_test = 0
        for k in range(n_folds):
            lo, hi = int(edges[k]), int(edges[k + 1])
            train_end = lo - h  # train on positions p < lo - h
            if train_end < 100 or hi <= lo:
                continue
            tr = D[(D["date"] < dates[train_end]) & D[col].notna()]
            te = D[(D["date"] >= dates[lo]) & (D["date"] <= dates[hi - 1]) & D[col].notna()]
            if len(tr) < 500 or len(te) < 100:
                continue
            model = make_regressor().fit(tr[feature_cols], tr[col])
            pred = model.predict(te[feature_cols])
            real = te[col].to_numpy()
            ic = pd.Series(pred).corr(pd.Series(real), method="spearman")
            ics.append(0.0 if pd.isna(ic) else float(ic))
            hits.append(float((np.sign(pred) == np.sign(real)).mean()))
            ups.append(float((real > 0).mean()))
            n_test += len(te)

        if ics:
            ic_mean = float(np.mean(ics))
            pos_frac = float(np.mean([i > 0 for i in ics]))
        else:
            ic_mean, pos_frac = float("nan"), 0.0
        t_stat = float("nan")
        if len(ics) >= 2 and np.std(ics, ddof=1) > 0:
            t_stat = ic_mean / (np.std(ics, ddof=1) / np.sqrt(len(ics)))
        trusted = bool(
            len(ics) >= MIN_FOLDS and ic_mean >= MIN_IC and t_stat >= MIN_T_STAT
        )
        out[h] = {
            "ic_mean": ic_mean,
            "ic_by_fold": [round(i, 4) for i in ics],
            "t_stat": t_stat,
            "pos_frac": pos_frac,
            "hit_rate": float(np.mean(hits)) if hits else float("nan"),
            "base_up": float(np.mean(ups)) if ups else float("nan"),
            "n_test": n_test,
            "trusted": trusted,
        }
    return out


def trusted_horizons(metrics: dict[int, dict]) -> list[int]:
    return sorted(h for h, m in metrics.items() if m["trusted"])


def print_validation(metrics: dict[int, dict]) -> None:
    print("\n📊 Walk-forward validation (purged, expanding window)")
    print(f"   gate: mean IC >= {MIN_IC} and t-stat across folds >= {MIN_T_STAT}")
    print(f"   {'h':>3} {'IC':>8} {'t':>6} {'folds>0':>8} {'hit%':>6} {'up%':>6} {'n_test':>8}  trusted")
    for h in HORIZONS:
        m = metrics.get(h)
        if not m:
            continue
        print(
            f"   {h:>3} {m['ic_mean']:>8.4f} {m['t_stat']:>6.2f} {m['pos_frac']:>8.0%} {m['hit_rate']:>6.1%} "
            f"{m['base_up']:>6.1%} {m['n_test']:>8}  {'YES' if m['trusted'] else 'no'}"
        )
    t = trusted_horizons(metrics)
    print(f"   -> trusted horizons: {t if t else 'NONE (timing output will be withheld)'}\n")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_curve(models: dict[int, XGBRegressor], X: pd.DataFrame) -> dict[int, float]:
    """Expected cumulative % move at each horizon for a single-row X."""
    return {h: float(models[h].predict(X)[0]) for h in sorted(models)}


def add_trading_days(start, n: int) -> pd.Timestamp:
    """The date n trading days after `start` (start itself is day 0)."""
    start = pd.Timestamp(start)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    start = start.normalize()
    if mcal is not None:
        days = mcal.get_calendar("NYSE").valid_days(
            start_date=start + pd.Timedelta(days=1),
            end_date=start + pd.Timedelta(days=2 * n + 10),
        )
        return days[n - 1].tz_localize(None).normalize()
    return start + pd.offsets.BDay(n)


def summarize_curve(curve: dict[int, float], as_of, reach_frac: float = 0.8) -> dict:
    """Turn a delta curve into 'when' estimates.

    `as_of` is the date of the latest price bar the features came from (NOT
    today: on a weekend the latest bar is Friday).
    """
    hs = sorted(curve)
    mu = np.array([curve[h] for h in hs], dtype=float)

    # Direction follows the headline (base-horizon) signal when available.
    sign = 1.0 if curve.get(BASE_HORIZON, mu[0]) >= 0 else -1.0
    signed = sign * mu

    i_peak = int(np.argmax(signed))
    i_reach = int(np.argmax(signed >= reach_frac * signed[i_peak]))
    peak_h, reach_h = hs[i_peak], hs[i_reach]

    return {
        "reliable": True,
        "direction": "up" if sign > 0 else "down",
        "horizons_used": hs,
        "peak_horizon": peak_h,
        "peak_delta_pct": round(float(mu[i_peak]), 3),
        "peak_date": add_trading_days(as_of, peak_h).date().isoformat(),
        "reach_horizon": reach_h,
        "reach_date": add_trading_days(as_of, reach_h).date().isoformat(),
        # True => still rising at the longest usable horizon: read as "no earlier than".
        "peak_at_window_edge": i_peak == len(hs) - 1,
        "curve": {h: round(float(v), 3) for h, v in zip(hs, mu)},
    }


def build_manifest(curve: dict[int, float], trusted: list[int], as_of) -> dict:
    """Timing estimate using only horizons that passed validation (if REQUIRE_SKILL)."""
    used = sorted(trusted) if REQUIRE_SKILL else sorted(curve)
    used = [h for h in used if h in curve]
    if len(used) < MIN_TRUSTED_HORIZONS:
        return {
            "reliable": False,
            "reason": (
                f"only {len(used)} horizon(s) passed out-of-sample validation "
                f"(need {MIN_TRUSTED_HORIZONS})"
            ),
        }
    return summarize_curve({h: curve[h] for h in used}, as_of)