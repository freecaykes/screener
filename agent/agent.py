import asyncio
import os
from enum import Enum

import pandas as pd
import pandas_ta as ta
import yfinance as yf
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict, Optional, Any

import train.train


class Signal(Enum):
    STRONG_SELL = 0,
    SELL = 1,
    HOLD = 2,
    BUY = 3,
    STRONG_BUY = 4


class AgentState(TypedDict):
    ticker: str
    news: list[dict]
    headlines: list[str]
    price_data: Optional[pd.DataFrame]
    indicators: dict
    sentiment_score: float
    predicted_delta: float
    signal: str
    signal_confidence: float
    signal_score: float


class TickerAgent:
    llm: BaseChatModel
    workflow: StateGraph
    app: CompiledStateGraph[AgentState, Any, Any, Any]
    newsLimit: int

    def __init__(
            self,
            model: str,
            temp: float,
            newsLimit: int = 5
    ):
        print(f"🤖 Initializing TickerAgent for model {model}...")
        api_key = os.getenv("API_KEY")
        provider = os.getenv("LLM_PROVIDER")
        print(f"   Provider: {provider}")

        self.newsLimit = newsLimit

        # Determine the correct API key parameter based on provider
        kwargs = {
            "model": model,
            "temperature": temp,
        }

        if provider:
            kwargs["model_provider"] = provider

        if provider == "google_genai":
            kwargs["google_api_key"] = api_key
        elif provider == "openai":
            kwargs["api_key"] = api_key
        else:
            # Fallback
            kwargs["api_key"] = api_key

        print(f"   init_chat_model kwargs: { {k: v for k, v in kwargs.items() if 'key' not in k} }")
        self.llm = init_chat_model(**kwargs)

        workflow = StateGraph(state_schema=AgentState)

        workflow.add_node("fetch_news", self._fetch_news)
        workflow.add_node("extract_headline", self._extract_headline)
        workflow.add_node("compute_indicators", self._compute_indicators)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis)
        workflow.add_node("xgboost_predict", self._xgboost_predict)
        workflow.add_node("generate_signal", self._generate_signal)

        workflow.add_edge(START, "fetch_news")
        workflow.add_edge("fetch_news", "extract_headline")
        workflow.add_edge("extract_headline", "compute_indicators")
        workflow.add_edge("compute_indicators", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "xgboost_predict")
        workflow.add_edge("xgboost_predict", "generate_signal")
        workflow.add_edge("generate_signal", END)

        self.workflow = workflow
        self.app = self.workflow.compile()
        print("✅ TickerAgent workflow compiled.")

    async def run(self, ticker: str) -> AgentState:
        print(f"🚀 Running agent for {ticker}...")
        initial_state: AgentState = {"ticker": ticker}
        try:
            result = await self.app.ainvoke(initial_state)
            print(f"🏁 Finished agent for {ticker}.")
            return result
        except Exception as e:
            print(f"💥 Error in TickerAgent.run for {ticker}: {e}")
            raise

    async def _fetch_news(self, state: AgentState) -> AgentState:
        print(f"  [node] fetching news for {state['ticker']}...")
        ticker_obj = yf.Ticker(state["ticker"])
        raw_news = ticker_obj.news[:5]
        state["news"] = raw_news
        return state

    async def _extract_headline(self, state: AgentState) -> AgentState:
        # TODO fix news for multiple sources
        print(f"  [node] extracting headline for {state['ticker']}...")
        state["headlines"] = []
        if state["news"]:
            for news in state["news"]:
                state["headlines"].append(str(news.get("content").get("summary").strip()))

        return state

    async def _compute_indicators(self, state: AgentState) -> AgentState:
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

    async def _sentiment_analysis(self, state: AgentState) -> AgentState:
        print(f"  [node] sentiment analysis for {state['ticker']}...")
        if not state.get("headlines") or len(state["headlines"]) == 0:
            state["sentiment_score"] = 0.0
            return state

        prompt = f"""
        You are a professional financial sentiment analyst.
        Analyze ONLY the impact of these headlines separated by ',' on the stock price of {state["ticker"]}.
        Return a single number between -1.0 (strongly negative) and +1.0 (strongly positive).
        Do not explain — just the number.
    
        Headlines: {str(','.join(state["headlines"])).strip()}
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        try:
            score = float("".join(c for c in content if c.isdigit() or c in ".-"))
            score = max(min(score, 1.0), -1.0)
        except ValueError:
            score = 0.0

        state["sentiment_score"] = score
        return state

    async def _xgboost_predict(self, state: AgentState) -> AgentState:
        print(f"  [node] XGBoost prediction for {state['ticker']}...")
        if train.train.MODEL is None:
            state["predicted_delta"] = 0.0
            return state

        ind = state["indicators"]
        # Use a list to ensure order matches training exactly
        feature_cols = [
            "sentiment_score",
            "RSI_14",
            "price_to_ema21",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "BBB_20_2.0",
            "BBM_20_2.0"
        ]

        feat_values = {
            "sentiment_score": state["sentiment_score"],
            "RSI_14": ind.get("RSI_14", 50.0),
            "price_to_ema21": ind.get("price_to_ema21", 1.0),
            "MACD_12_26_9": ind.get("MACD_12_26_9", 0.0),
            "MACDs_12_26_9": ind.get("MACDs_12_26_9", 0.0),
            "BBB_20_2.0": ind.get("BBB_20_2.0", 0.02),
            "BBM_20_2.0": ind.get("BBM_20_2.0", 100.0),
        }

        def _predict():
            # Create DataFrame with explicit column order
            X = pd.DataFrame([[feat_values[col] for col in feature_cols]], columns=feature_cols)
            return train.train.MODEL.predict(X)[0]

        pred = await asyncio.to_thread(_predict)
        state["predicted_delta"] = round(float(pred), 4)
        return state

    async def _generate_signal(self, state: AgentState) -> AgentState:
        """
        Clean weighted scoring system for generating trading signals.
        """
        delta = state["predicted_delta"]  # XGBoost predicted % move
        sentiment = state["sentiment_score"]
        vix = state["indicators"].get("vix_current", 20.0)
        pullback = bool(state["indicators"].get("pullback_buy_setup", 0))

        # ====================== WEIGHTED SCORING ======================
        score = 0.0

        # Core components with tunable weights
        score += delta * 0.50  # XGBoost prediction has highest weight
        score += sentiment * 0.30  # Sentiment from LLM
        score += (1 if pullback else -0.4) * 0.15  # Pullback setup bonus/penalty
        score -= (vix - 18) * 0.012  # High VIX penalty (fear reduces conviction)

        # Optional: Add momentum bonus
        rsi = state["indicators"].get("RSI_14", 50)
        if 45 < rsi < 65:  # Healthy momentum zone during pullback
            score += 0.25

        # ====================== SIGNAL MAPPING ======================
        if score >= 1.85:
            signal = "STRONG BUY"
            confidence = min(0.92, 0.45 + score * 0.18)
        elif score >= 0.95:
            signal = "BUY"
            confidence = min(0.82, 0.40 + score * 0.22)
        elif score >= 0.25:
            signal = "BUY"
            confidence = min(0.68, 0.35 + score * 0.20)
        elif score >= -0.35:
            signal = "HOLD"
            confidence = 0.75
        elif score >= -1.1:
            signal = "SELL"
            confidence = min(0.78, 0.45 + abs(score) * 0.22)
        else:
            signal = "STRONG SELL"
            confidence = min(0.90, 0.50 + abs(score) * 0.20)

        state["signal"] = signal
        state["signal_confidence"] = round(float(confidence), 2)
        state["signal_score"] = round(float(score), 3)  # Useful for debugging

        return state
