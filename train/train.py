# =============================================================================
# TRAIN / LOAD MODEL (same as before)
# =============================================================================
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from xgboost import XGBRegressor

MODEL: Optional[XGBRegressor] = None

def xgboost(tickers: list[str]) -> XGBRegressor:
    global MODEL
    model_path = "xgboost_stock_delta_model.pkl"

    if os.path.exists(model_path):
        print(f"✅ Loading existing model from {model_path}")
        MODEL = joblib.load(model_path)
        return MODEL

    print("🚀 Training new XGBoost model...")

    all_X = []
    all_y = []

    for tkr in tickers:
        print(f"   → {tkr}")
        df = yf.download(tkr, period="2y", progress=False)
        if len(df) < 100:
            continue

        # Flatten MultiIndex if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # === INDICATORS ===
        # Ensure Close is a Series
        close_series = df["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        df["RSI_14"] = ta.rsi(close_series, length=14)

        ema21 = ta.ema(close_series, length=21)
        if isinstance(ema21, pd.DataFrame):
            ema21 = ema21.iloc[:, 0]
        df["EMA_21"] = ema21

        df["price_to_ema21"] = (close_series / df["EMA_21"]).astype('float32')

        macd = ta.macd(close_series, fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            df = pd.concat([df, macd], axis=1)

        bb = ta.bbands(close_series, length=20, std=2)
        if isinstance(bb, pd.DataFrame) and not bb.empty:
            df = pd.concat([df, bb], axis=1)
            # Rename columns to standard names for easier access
            bb_map = {
                col: "BBB_20_2.0" for col in bb.columns if col.startswith("BBB_")
            }
            bb_map.update({
                col: "BBM_20_2.0" for col in bb.columns if col.startswith("BBM_")
            })
            df = df.rename(columns=bb_map)

        # Drop NaNs after all indicators are computed
        df = df.dropna()

        # Dummy features (only sentiment_score)
        df["sentiment_score"] = np.random.uniform(-1.0, 1.0, len(df))

        df["target_delta"] = (close_series.shift(-1) - close_series) / close_series * 100
        df = df.dropna().reset_index(drop=True)

        # Force numeric for all columns we might use
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')

        # Feature columns - EXPLICIT LIST
        feature_cols = [
            "sentiment_score",
            "RSI_14",
            "price_to_ema21",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "BBB_20_2.0",
            "BBM_20_2.0"
        ]

        # Verify all columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️ Missing columns for {tkr}: {missing_cols}")
            continue

        X = df[feature_cols].copy()
        y = df["target_delta"].copy()

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No training data collected")

    X_total = pd.concat(all_X, ignore_index=True)
    y_total = pd.concat(all_y, ignore_index=True)

    MODEL = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    MODEL.fit(X_total, y_total)

    joblib.dump(MODEL, model_path)
    print(f"✅ Model trained successfully with {X_total.shape[1]} features")
    return MODEL
