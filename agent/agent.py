import asyncio
import os
import re
from enum import Enum

import pandas as pd
import yfinance as yf
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict, Optional, Any

from . import decision, newsfeed
from train import features, timing, train


NEWS_FETCH_TIMEOUT_SEC = 30


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
    as_of: str                  # date of the latest bar the features were computed from
    sentiment_score: float
    predicted_delta: float      # XGBoost expected % move over timing.BASE_HORIZON trading days
    delta_curve: dict           # {horizon_days: expected cumulative % move}
    manifest: dict              # when the move is expected to play out (see timing.build_manifest)
    model_validated: bool       # did the base-horizon model show out-of-sample skill?
    signal: str
    signal_confidence: float
    signal_score: float         # expected move in sigma units (signed)
    signal_detail: dict         # why: components, weights and their provenance


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
        print(f"   [node] fetching news for {state['ticker']}...")
        try:
            state["news"] = await asyncio.wait_for(
                asyncio.to_thread(newsfeed.fetch_news, state["ticker"], self.newsLimit),
                timeout=NEWS_FETCH_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            print(f"news fetch timed out after {NEWS_FETCH_TIMEOUT_SEC}s for {state['ticker']}")
            state["news"] = []
        except Exception as e:  # keep the pipeline alive: no news just means sentiment = 0
            print(f"news fetch failed for {state['ticker']}: {type(e).__name__}: {e}")
            state["news"] = []
        if not state["news"]:
            print(f"no news items for {state['ticker']}")
        return state

    async def _extract_headline(self, state: AgentState) -> AgentState:
        print(f"   [node] extracting headline for {state['ticker']}...")
        state["headlines"] = []
        for item in state.get("news") or []:
            n = newsfeed.normalize(item)
            text = n["summary"] or n["title"]
            if text:
                state["headlines"].append(text)
        return state

    async def _compute_indicators(self, state: AgentState) -> AgentState:
        # Uses train/features.py -- the exact code the model was trained with.
        # Blocking yfinance calls run in threads so they don't stall the event loop.
        ticker = state["ticker"]
        prices, vix = await asyncio.gather(
            asyncio.to_thread(features.fetch_history, ticker, features.LIVE_PERIOD),
            asyncio.to_thread(features.fetch_vix, features.LIVE_PERIOD),
        )
        feats = features.build_features(prices, vix)

        if feats.empty:
            print(f"no usable features for {ticker} (price/VIX data missing or too short)")
            state["price_data"] = None
            state["indicators"] = {}
            return state

        latest = feats.iloc[-1]
        state["price_data"] = prices
        state["as_of"] = feats.index[-1].date().isoformat()
        state["indicators"] = {c: float(latest[c]) for c in features.FEATURE_COLS}
        return state

    async def _sentiment_analysis(self, state: AgentState) -> AgentState:
        print(f"   [node] sentiment analysis for {state['ticker']}...")
        headlines = str(','.join(state["headlines"])).strip() if state.get("headlines") or len(state["headlines"]) > 0 else ""

        # NOTE: the prompt gets the indicator dict, not the whole `state` (which now
        # holds a year of OHLCV and the raw news payloads).
        indicators = state.get("indicators", {})
        prompt = f"""
        Analyze ONLY the impact of these headlines separated by ',' on the stock price of {state["ticker"]} given the
        following indicator values {indicators}
        Return a single number between -1.0 (strongly negative) and +1.0 (strongly positive).
        Given the current VIX indicator is at {indicators.get("vix_current", "unknown")}
        Headlines: {headlines}

        If there are no headlines provided, give a sentiment score generated from the most recent top news headline.
        Output the one headline from either one of the provided headlines provided or on that you found that has the largest weight on the sentiment score with the score itself in this format
        
        headline: <headline name>
        score: <sentiment score>

        eg.
        headline: "AMD Stock Is Set to Benefit From Both Physical and Agentic AI"
        score: 0.65
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        content = self._response_text(response)
        data = dict(re.findall(r"([A-Za-z]+):\s*([^\n]+)", content))

        headline = data.get("headline")
        score = max(min(float(data.get("score")), 1.0), -1.0)

        state["sentiment_score"] = score
        state["headlines"] = [headline]
        return state

    @staticmethod
    def _response_text(response) -> str:
        """LangChain returns .content as a str for some providers and a list of blocks for others."""
        content = response.content
        if isinstance(content, str):
            return content.strip()
        parts = []
        for block in content or []:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return " ".join(parts).strip()

    async def _xgboost_predict(self, state: AgentState) -> AgentState:
        print("_xgboost_predict")
        state["predicted_delta"] = 0.0
        state["delta_curve"] = {}
        state["model_validated"] = False
        state["manifest"] = {"reliable": False, "reason": "no model or no indicators"}

        bundle = train.get_bundle()
        ind = state.get("indicators") or {}
        if bundle is None or not ind:
            return state

        # Sentiment is deliberately NOT a model input (no historical sentiment exists to
        # train on). It enters the decision in _generate_signal instead.
        cols = features.FEATURE_COLS
        X = pd.DataFrame([{c: float(ind[c]) for c in cols}], columns=cols)

        curve = timing.predict_curve(bundle["models"], X)
        state["delta_curve"] = {h: round(v, 4) for h, v in curve.items()}
        state["predicted_delta"] = round(curve[timing.BASE_HORIZON], 4)
        state["model_validated"] = timing.BASE_HORIZON in bundle["trusted"]
        state["manifest"] = timing.build_manifest(curve, bundle["trusted"], pd.Timestamp(state["as_of"]))

        print(f"DEBUG [{state['ticker']}] {timing.BASE_HORIZON}d Δ: {state['predicted_delta']:+.4f}% "
              f"(validated={state['model_validated']})")
        return state

    async def _generate_signal(self, state: AgentState) -> AgentState:
        """Expected-value decision (see agent/decision.py).

        Replaces the old hand-weighted sum. The model's forecast is scaled by its measured
        out-of-sample calibration, LLM sentiment by a weight learned from this agent's own
        logged calls, both in units of the ticker's current volatility; confidence is the
        probability the move beats costs.
        """
        print("_generate_signal")
        ind = state.get("indicators") or {}
        bundle = train.get_bundle()
        base_metrics = ((bundle or {}).get("metrics") or {}).get(timing.BASE_HORIZON)

        out = decision.decide(
            pred_delta=state.get("predicted_delta", 0.0),
            base_metrics=base_metrics,
            vol_20d=ind.get("vol_20d"),
            sentiment=state.get("sentiment_score", 0.0),
            params=decision.load_params(),
            as_of=state.get("as_of"),
            manifest=state.get("manifest"),
        )
        state["signal"] = out["signal"]
        state["signal_confidence"] = out["confidence"]
        state["signal_score"] = out["score"]
        state["signal_detail"] = out["detail"]

        decision.record(state, out)  # shadow log: this is what lets sentiment's weight be learned
        return state