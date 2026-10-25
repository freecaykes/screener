# =============================================================================
# TRAIN / LOAD THE SETUP MODEL
# =============================================================================
# The app trades four swing setups on the tickers YOU list (nothing else is trained on):
#   pullback_buy / breakout_buy  and their mirrors  rally_sell / breakdown_sell.
# This module learns, from those tickers' own history, how likely each setup is to reach its
# profit target before its stop (train/setups.py defines both), and checks honestly whether the
# model has any out-of-sample skill (train/setup_model.py). SPY and the VIX are downloaded as
# reference series only; they are never training rows.
#
# The cache stores the feature list, strategy parameters, requested tickers and benchmarks, so a
# stale pickle is retrained instead of silently loaded.
# =============================================================================
import os
from typing import Optional

import joblib
import pandas as pd

from train import features, setup_model, setups

TRAIN_PERIOD = "10y"
MODEL_PATH = "setup_bundle_v1.pkl"
BUNDLE_VERSION = 1

BUNDLE: Optional[dict] = None


def _strategy_signature() -> dict:
    return {
        "tp_atr": setups.TP_ATR, "sl_atr": setups.SL_ATR, "max_hold": setups.MAX_HOLD, "cost_pct": setups.COST_PCT,
        "pullback_min_atr": setups.PULLBACK_MIN_ATR, "breakout_lookback": setups.BREAKOUT_LOOKBACK,
        "breakout_min_rel_volume": setups.BREAKOUT_MIN_REL_VOLUME, "ema_touch_atr": setups.EMA_TOUCH_ATR,
    }


def get_bundle() -> Optional[dict]:
    return BUNDLE


def get_model():
    return BUNDLE["model"] if BUNDLE else None


def xgboost(tickers: list[str], force_retrain: bool = False) -> Optional[dict]:
    """Load the cached setup model for `tickers`, or train one. Returns the bundle.

    If there are too few labelled setups to learn from, the bundle has model=None: setups are still
    detected and reported (they are rules, not predictions), but no probability is produced.
    """
    global BUNDLE
    tickers = list(dict.fromkeys(tickers))
    want_benchmarks = {t: features.benchmark_for(t) for t in sorted(set(tickers))}

    if not force_retrain and os.path.exists(MODEL_PATH):
        cached = joblib.load(MODEL_PATH)
        fresh = (
            cached.get("version") == BUNDLE_VERSION
            and cached.get("feature_cols") == setups.SETUP_FEATURES
            and cached.get("strategy") == _strategy_signature()
            and cached.get("requested") == sorted(set(tickers))
            and cached.get("benchmarks") == want_benchmarks
        )
        if fresh:
            BUNDLE = cached
            print(f"✅ Loaded setup model from {MODEL_PATH} (trained through {cached['trained_through']})")
            setup_model.print_validation(cached["metrics"], cached["rules"], cached["baselines"])
            return BUNDLE
        print(f"⚠️  {MODEL_PATH} is stale (features, strategy, tickers or benchmarks changed) -> retraining")

    print("🚀 Building swing-setup dataset from your tickers (market-relative features, shared with live inference)...")
    D, A = setup_model.build_dataset(tickers, TRAIN_PERIOD)
    baselines = setup_model.baselines_from(A)
    print(f"   {len(D)} labelled setups across {D['ticker'].nunique()} ticker(s), "
          f"{D['date'].min().date()} → {D['date'].max().date()}")

    metrics = setup_model.walk_forward_validate(D)
    rules = setup_model.rule_validation(D, A)
    setup_model.print_validation(metrics, rules, baselines)

    if len(D) < setup_model.MIN_ROWS_TO_FIT:
        print(f"⚠️  only {len(D)} labelled setups (< {setup_model.MIN_ROWS_TO_FIT}): not enough to learn from. "
              f"Setups will still be detected and reported, without a model probability. "
              f"Adding tickers is the only way to get more.")
        model = None
    else:
        model = setup_model.fit_classifier(D[setups.SETUP_FEATURES], D["y"])

    BUNDLE = {
        "version": BUNDLE_VERSION,
        "feature_cols": list(setups.SETUP_FEATURES),
        "strategy": _strategy_signature(),
        "model": model,
        "metrics": metrics,
        "rules": rules,
        "baselines": baselines,
        "requested": sorted(set(tickers)),
        "benchmarks": want_benchmarks,
        "trained_on": sorted(D["ticker"].unique().tolist()),
        "trained_through": str(D["date"].max().date()),
    }
    joblib.dump(BUNDLE, MODEL_PATH)
    print(f"✅ Saved to {MODEL_PATH}" + ("" if model is not None else "  (no model: too few setups)"))
    return BUNDLE


def flatten_columns(df):
    """Kept for backwards compatibility (other modules may import it)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df