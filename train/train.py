# =============================================================================
# TRAIN / LOAD MODEL
# =============================================================================
# Changes vs the previous version:
#   * sentiment_score is NOT a model feature any more (it was np.random noise in
#     training). It still feeds the signal formula in agent._generate_signal.
#   * Features come from train/features.py -- the same code the agent runs live,
#     so train/serve skew (pullback definition, MACD units, ...) is gone.
#   * Scale-dependent features (raw EMA_21, raw MACD) replaced by ratios.
#   * Trains ONLY on the tickers passed to xgboost() with 10y of history (was 2y),
#     one model per horizon, with purged walk-forward validation whose result
#     gates the timing output.
#   * No more `.fillna(0.0)` on indicators (that turned warm-up NaNs into fake
#     RSI = 0 rows).
#   * The cache file is versioned and stores the feature list AND the requested
#     tickers, so a stale pickle (or one trained on different tickers) is retrained
#     instead of silently loaded.
# =============================================================================
import os
from typing import Optional

import joblib
import pandas as pd
from xgboost import XGBRegressor

from train import features, timing

TRAIN_PERIOD = "10y"
MIN_ROWS_PER_TICKER = 300

MODEL_PATH = "xgboost_bundle_v2.pkl"
BUNDLE_VERSION = 2

MODEL: Optional[XGBRegressor] = None   # the BASE_HORIZON model (what `predicted_delta` uses)
BUNDLE: Optional[dict] = None          # models for all horizons + validation metrics


def get_model() -> Optional[XGBRegressor]:
    return MODEL


def get_bundle() -> Optional[dict]:
    return BUNDLE


def get_horizon_models() -> dict:
    return BUNDLE["models"] if BUNDLE else {}


def build_dataset(tickers: list[str], period: Optional[str] = None) -> pd.DataFrame:
    """Pooled frame: date, ticker, FEATURE_COLS, target_delta_* (one row per ticker-day)."""
    period = period or TRAIN_PERIOD
    vix = features.fetch_vix(period)
    if vix.empty:
        raise RuntimeError("Could not download ^VIX; refusing to train without it.")

    frames = []
    for tkr in tickers:
        try:
            prices = features.fetch_history(tkr, period)
            feats = features.build_features(prices, vix)
        except Exception as e:  # network hiccup / bad symbol: skip, don't die
            print(f"  ⚠️  {tkr}: skipped ({e})")
            continue
        if len(feats) < MIN_ROWS_PER_TICKER:
            print(f"  ⚠️  {tkr}: only {len(feats)} usable rows, skipped")
            continue

        # Targets from the FULL close series so shift(-h) is real trading days.
        targets = timing.add_horizon_targets(prices[["Close"]].copy()).drop(columns="Close")
        feats = feats.join(targets)
        feats["ticker"] = tkr
        feats["date"] = feats.index
        frames.append(feats.reset_index(drop=True))
        print(f"  → {tkr}: {len(feats)} rows")

    if not frames:
        raise ValueError("No training data collected")
    return pd.concat(frames, ignore_index=True)


def xgboost(tickers: list[str], force_retrain: bool = False) -> XGBRegressor:
    global MODEL, BUNDLE

    if not force_retrain and os.path.exists(MODEL_PATH):
        cached = joblib.load(MODEL_PATH)
        fresh = (
            cached.get("version") == BUNDLE_VERSION
            and cached.get("feature_cols") == features.FEATURE_COLS
            and cached.get("horizons") == timing.HORIZONS
            and cached.get("requested") == sorted(set(tickers))
        )
        if fresh:
            BUNDLE = cached
            MODEL = cached["models"][timing.BASE_HORIZON]
            print(f"✅ Loaded model bundle from {MODEL_PATH} (trained through {cached['trained_through']})")
            timing.print_validation(cached["metrics"])
            return MODEL
        print(f"⚠️  {MODEL_PATH} is stale (features, horizons or tickers changed) -> retraining")

    print("🚀 Training multi-horizon XGBoost (shared features, no sentiment)...")
    D = build_dataset(list(dict.fromkeys(tickers)))
    print(f"   dataset: {len(D):,} rows, {D['ticker'].nunique()} tickers, "
          f"{D['date'].min().date()} → {D['date'].max().date()}")

    metrics = timing.walk_forward_validate(D, features.FEATURE_COLS)
    timing.print_validation(metrics)

    models = timing.fit_horizon_models(D, features.FEATURE_COLS)

    BUNDLE = {
        "version": BUNDLE_VERSION,
        "feature_cols": list(features.FEATURE_COLS),
        "horizons": list(timing.HORIZONS),
        "models": models,
        "metrics": metrics,
        "trusted": timing.trusted_horizons(metrics),
        "requested": sorted(set(tickers)),
        "trained_on": sorted(D["ticker"].unique().tolist()),
        "trained_through": str(D["date"].max().date()),
    }
    joblib.dump(BUNDLE, MODEL_PATH)
    MODEL = models[timing.BASE_HORIZON]
    print(f"✅ Trained {len(models)} horizon models on {len(features.FEATURE_COLS)} features. "
          f"Saved to {MODEL_PATH}")
    return MODEL


def flatten_columns(df):
    """Kept for backwards compatibility (other modules may import it)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df