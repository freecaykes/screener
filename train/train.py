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

        # === VIX - SAFER JOIN ===
        vix_df = yf.download("^VIX", period="2y", progress=False)

        # Flatten any MultiIndex columns (common cause of the error)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        vix_df = vix_df[['Close']].rename(columns={'Close': 'vix_current'})

        # Reset indexes to avoid level mismatch
        df = df.reset_index()
        vix_df = vix_df.reset_index()

        # Merge on Date
        df = df.merge(vix_df[['Date', 'vix_current']], on='Date', how='left')

        # Forward fill VIX (markets are closed on different days sometimes)
        df["vix_current"] = df["vix_current"].ffill()

        # Cleanup
        df = df.dropna().reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')

        # Dummy features (only sentiment_score)
        df["sentiment_score"] = np.random.uniform(-1.0, 1.0, len(df))
        df["pullback_buy_setup"] = ((df["Close"] > df["EMA_21"] * 0.97) &
                                    (df["Close"] < df["EMA_21"] * 1.03)).astype('float32')

        df["target_delta"] = (close_series.shift(-1) - close_series) / close_series * 100
        df = df.dropna().reset_index(drop=True)

        # Force numeric for all columns we might use
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')

        # Feature columns - EXPLICIT LIST
        feature_cols = [
            "sentiment_score",
            "pullback_buy_setup",
            "EMA_21"
            "RSI_14",
            "price_to_ema21",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "BBB_20_2.0",
            "BBM_20_2.0",
            "vix_current",
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
