# =============================================================================
# TRAIN / LOAD MODEL (same as before)
# =============================================================================
import os
from typing import Optional

import numpy as np
import joblib
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from xgboost import XGBRegressor

MODEL: Optional[XGBRegressor] = None


def get_model() -> Optional[XGBRegressor]:
    global MODEL
    return MODEL


def xgboost(tickers: list[str]) -> XGBRegressor:
    global MODEL
    model_path = "xgboost_stock_delta_model.pkl"

    if os.path.exists(model_path):
        print(f"✅ Loading existing model from {model_path}")
        MODEL = joblib.load(model_path)
        return MODEL

    print("🚀 Training XGBoost model with VIX (Ultra Safe Version)...")

    all_X = []
    all_y = []

    # Download VIX once
    vix_df = flatten_columns(yf.download("^VIX", period="2y", progress=False))

    # Force single level columns
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(1)

    vix_df = vix_df[['Close']].rename(columns={'Close': 'vix_current'})
    vix_df = vix_df.reset_index()  # Make sure it has 'Date' column
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])

    for tkr in tickers:
        print(f"   → {tkr}")
        df = flatten_columns(yf.download(tkr, period="2y", progress=False))
        if len(df) < 150:
            continue

        # === INDICATORS ===
        df["RSI_14"] = ta.rsi(df["Close"], length=14)

        ema21 = ta.ema(df["Close"], length=21)
        if isinstance(ema21, pd.DataFrame):
            ema21 = ema21.iloc[:, 0]
        df["EMA_21"] = ema21

        df = df.dropna()
        df["price_to_ema21"] = df["Close"] / df["EMA_21"]

        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            df = pd.concat([df, macd], axis=1)

        bb = ta.bbands(df["Close"], length=20, std=2)
        if isinstance(bb, pd.DataFrame) and not bb.empty:
            df = pd.concat([df, bb], axis=1)

        # === SAFE VIX ALIGNMENT (No join/merge issues) ===
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        # Merge with explicit column handling
        df = pd.merge(
            df,
            vix_df[['Date', 'vix_current']],
            on='Date',
            how='left'
        )

        df["vix_current"] = df["vix_current"].ffill()

        # Final cleanup
        df = df.dropna().reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')

        # Features
        df["sentiment_score"] = np.random.uniform(-1.0, 1.0, len(df))
        df["pullback_buy_setup"] = ((df["Close"] > df["EMA_21"] * 0.97) &
                                    (df["Close"] < df["EMA_21"] * 1.03)).astype('float32')

        df["target_delta"] = (df["Close"].shift(-3) - df["Close"]) / df["Close"] * 100
        df = df.dropna().reset_index(drop=True)

        feature_cols = [
            "sentiment_score",
            "pullback_buy_setup",
            "RSI_14",
            "price_to_ema21",
            "vix_current",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "EMA_21",
        ]

        X = df[feature_cols].copy()
        y = df["target_delta"].copy()

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No training data collected")

    X_total = pd.concat(all_X, ignore_index=True)
    y_total = pd.concat(all_y, ignore_index=True)

    MODEL = XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    MODEL.fit(X_total, y_total)

    joblib.dump(MODEL, model_path)
    print(f"✅ Model trained successfully! Features: {X_total.shape[1]}")
    return MODEL

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        # drop whichever level is the ticker/constant level
        df.columns = df.columns.get_level_values(0)
    return df