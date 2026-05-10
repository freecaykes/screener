# =============================================================================
# MULTI-TICKER CONTINUOUS LANGGRAPH AGENT WITH XGBoost DELTA PREDICTOR
# =============================================================================
# pip install yfinance langgraph langchain langchain-openai pandas pandas_ta xgboost joblib
#
# Set your API key:
# export OPENAI_API_KEY=sk-...
#
# This script:
#   • Runs MULTIPLE tickers IN PARALLEL (each in its own thread)
#   • Continuously polls Yahoo Finance for the LATEST news every POLL_INTERVAL_SEC seconds
#   • Only triggers full analysis (sentiment + indicators + XGBoost) when a BRAND NEW headline appears
#   • Replaces the simple rule-based predictor with a trained XGBoost regressor
#   • XGBoost is trained once at startup on 2 years of historical data for all tickers
#     (dummy sentiment is used for demo; see comment inside train_or_load_xgboost for real-world upgrade)
#
# Run with: python stock_news_multi_agent.py
# Press Ctrl+C to stop all agents gracefully.
# =============================================================================

import os
import time
import threading
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import joblib
from typing import TypedDict, Optional
from xgboost import XGBRegressor

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# =============================================================================
# CONFIGURATION
# =============================================================================
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # or hard-code for testing

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]  # ← Edit this list for your stocks
POLL_INTERVAL_SEC = 300  # 5 minutes – good balance (news isn't instant; avoids rate limits & cost)
NEWS_LIMIT = 5

# Global XGBoost model (shared across all ticker agents)
MODEL: Optional[XGBRegressor] = None


# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    ticker: str
    news: list[dict]
    headline: str
    price_data: Optional[pd.DataFrame]
    indicators: dict
    sentiment_score: float
    predicted_delta: float


# =============================================================================
# NODE 1: FETCH NEWS
# =============================================================================
def fetch_news(state: AgentState) -> AgentState:
    ticker_obj = yf.Ticker(state["ticker"])
    raw_news = ticker_obj.news[:NEWS_LIMIT]
    state["news"] = raw_news
    return state


# =============================================================================
# NODE 2: EXTRACT HEADLINE
# =============================================================================
def extract_headline(state: AgentState) -> AgentState:
    if state["news"]:
        headline = state["news"][0].get("title") or state["news"][0].get("content", "No headline found")
        state["headline"] = headline.strip()
    else:
        state["headline"] = "No recent news found"
    return state


# =============================================================================
# NODE 3: COMPUTE TECHNICAL INDICATORS (exact columns used by XGBoost)
# =============================================================================
def compute_indicators(state: AgentState) -> AgentState:
    ticker_obj = yf.Ticker(state["ticker"])
    df: pd.DataFrame = ticker_obj.history(period="60d")
    if df.empty:
        state["price_data"] = None
        state["indicators"] = {}
        return state

    # Trend + Momentum + Volatility (same as training)
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    bb = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    latest = df.iloc[-1]

    state["price_data"] = df
    state["indicators"] = {
        "RSI_14": float(latest["RSI_14"]) if pd.notna(latest["RSI_14"]) else 50.0,
        "MACD_12_26_9": float(latest["MACD_12_26_9"]) if pd.notna(latest.get("MACD_12_26_9")) else 0.0,
        "MACDs_12_26_9": float(latest.get("MACDs_12_26_9", 0)) if pd.notna(latest.get("MACDs_12_26_9")) else 0.0,
        "BBM_20_2.0": float(latest["BBM_20_2.0"]) if pd.notna(latest.get("BBM_20_2.0")) else 100.0,
        "BBB_20_2.0": float(latest["BBB_20_2.0"]) if pd.notna(latest.get("BBB_20_2.0")) else 0.02,
    }
    return state


# =============================================================================
# NODE 4: LLM SENTIMENT (same as before)
# =============================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def sentiment_analysis(state: AgentState) -> AgentState:
    if not state.get("headline") or state["headline"] == "No recent news found":
        state["sentiment_score"] = 0.0
        return state

    prompt = f"""
    You are a professional financial sentiment analyst.
    Analyze ONLY the impact of this headline on the stock price of {state["ticker"]}.
    Return a single number between -1.0 (strongly negative) and +1.0 (strongly positive).
    Do not explain, do not add any text — just the number.

    Headline: {state["headline"]}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    try:
        score = float("".join(c for c in content if c.isdigit() or c in ".-"))
        score = max(min(score, 1.0), -1.0)
    except ValueError:
        score = 0.0

    state["sentiment_score"] = score
    return state


# =============================================================================
# NODE 5: XGBoost DELTA PREDICTOR (replaces the old rule-based predictor)
# =============================================================================
def xgboost_predict(state: AgentState) -> AgentState:
    global MODEL
    if MODEL is None:
        state["predicted_delta"] = 0.0
        return state

    ind = state["indicators"]
    feat_dict = {
        "sentiment_score": state["sentiment_score"],
        "RSI_14": ind.get("RSI_14", 50.0),
        "MACD_12_26_9": ind.get("MACD_12_26_9", 0.0),
        "MACDs_12_26_9": ind.get("MACDs_12_26_9", 0.0),
        "BBB_20_2.0": ind.get("BBB_20_2.0", 0.02),
        "BBM_20_2.0": ind.get("BBM_20_2.0", 100.0),
    }

    X = pd.DataFrame([feat_dict])
    pred = MODEL.predict(X)[0]
    state["predicted_delta"] = round(float(pred), 4)
    return state


# =============================================================================
# BUILD THE LANGGRAPH WORKFLOW (one shared graph for all tickers)
# =============================================================================
workflow = StateGraph(AgentState)

workflow.add_node("fetch_news", fetch_news)
workflow.add_node("extract_headline", extract_headline)
workflow.add_node("compute_indicators", compute_indicators)
workflow.add_node("sentiment_analysis", sentiment_analysis)
workflow.add_node("xgboost_predict", xgboost_predict)

workflow.add_edge(START, "fetch_news")
workflow.add_edge("fetch_news", "extract_headline")
workflow.add_edge("extract_headline", "compute_indicators")
workflow.add_edge("compute_indicators", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "xgboost_predict")
workflow.add_edge("xgboost_predict", END)

app = workflow.compile()


# =============================================================================
# TRAIN / LOAD XGBoost MODEL (runs once at startup)
# =============================================================================
def load_or_train_xgboost(tickers: list[str]) -> XGBRegressor:
    global MODEL
    model_path = "xgboost_stock_delta_model.pkl"

    if os.path.exists(model_path):
        print(f"✅ Loading existing XGBoost model from {model_path}")
        return joblib.load(model_path)

    print("🚀 Training new XGBoost model on 2 years of historical data...")
    all_X = []
    all_y = []

    for tkr in tickers:
        print(f"   → Collecting data for {tkr}")
        df = yf.download(tkr, period="2y", progress=False)
        if len(df) < 100:
            continue

        # Replicate exact indicator calculation used in the agent
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        bb = ta.bbands(df["Close"], length=20, std=2)
        df = pd.concat([df, bb], axis=1)

        df = df.dropna()

        # Dummy sentiment for demo (replace this in production!)
        # TODO: For REAL accuracy, run the same LLM on historical news archives
        # and store the actual sentiment_score for each past day.
        df["sentiment_score"] = np.random.uniform(-1.0, 1.0, len(df))

        # Target = next-day percentage move (what we want to predict)
        df["target_delta"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"] * 100
        df = df.dropna()

        feature_cols = ["sentiment_score", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "BBB_20_2.0", "BBM_20_2.0"]
        X = df[feature_cols]
        y = df["target_delta"]

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No training data could be collected")

    X_total = pd.concat(all_X)
    y_total = pd.concat(all_y)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_total, y_total)

    joblib.dump(model, model_path)
    print(f"✅ XGBoost model trained and saved to {model_path}")
    return model


# =============================================================================
# CONTINUOUS WORKER (runs in parallel per ticker)
# =============================================================================
def continuous_worker(ticker: str):
    global MODEL
    last_headline = None

    print(f"📡 Started continuous agent for {ticker}")

    while True:
        try:
            initial_state: AgentState = {"ticker": ticker}
            result: AgentState = app.invoke(initial_state)

            current_headline = result.get("headline", "")

            # Only process and print when we see a BRAND NEW headline
            if current_headline and current_headline != last_headline and current_headline != "No recent news found":
                print(f"\n🔥 [NEW NEWS] {ticker} @ {time.strftime('%H:%M:%S')}")
                print(f"   Headline : {current_headline}")
                print(f"   Sentiment: {result['sentiment_score']:.2f}")
                print(f"   Indicators:")
                for k, v in result["indicators"].items():
                    print(f"      {k:15} → {v:.4f}")
                print(f"   🔮 XGBoost Predicted Delta: {result['predicted_delta']:+.2f}%")
                last_headline = current_headline

            # Optional: print "no change" every few polls (comment out if too noisy)
            # else:
            #     print(f"   {ticker} – no new news")

        except Exception as e:
            print(f"⚠️ Error in {ticker} worker: {e}")

        time.sleep(POLL_INTERVAL_SEC)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Train or load the shared XGBoost model once
    MODEL = load_or_train_xgboost(TICKERS)

    # Launch one continuous thread per ticker (true parallel execution)
    threads = []
    for ticker in TICKERS:
        t = threading.Thread(target=continuous_worker, args=(ticker,), daemon=True)
        t.start()
        threads.append(t)

    print(f"\n🚀 All {len(TICKERS)} LangGraph agents running in parallel!")
    print(f"   Polling every {POLL_INTERVAL_SEC} seconds for new Yahoo Finance news.")
    print("   Press Ctrl+C to stop.\n")

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n👋 Shutting down all agents... Goodbye!")