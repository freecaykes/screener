import asyncio
import json
import os
import re
from enum import Enum
import pandas as pd

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated*",
    module=r"yfinance",
)

import yfinance as yf
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict, Optional, Any

from agent import decision, newsfeed
from train import features, setup_model, setups, train


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
    indicators: dict            # setup features of the last COMPLETED session (+ Close)
    as_of: str                  # that session's date
    benchmark: str              # reference for the relative-strength features (e.g. SPY)
    setup: str                  # pullback_buy | breakout_buy | rally_sell | breakdown_sell | "" (none)
    p_info: Optional[dict]      # layered probability the trade hits its target before its stop
    news_analysis: Optional[dict]  # LLM reading of the headlines in the context of the setup
    sentiment_score: float      # news_analysis["sentiment"] (0 if none)
    signal: str
    signal_confidence: float    # P(target before stop) after news; 0 when abstaining before estimating
    signal_score: float         # expected R per trade after costs
    signal_detail: dict         # why: evidence, weights, plan (entry / stop / target / time stop)


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
        workflow.add_node("setup_probability", self._setup_probability)
        workflow.add_node("news_analysis", self._news_analysis)
        workflow.add_node("generate_signal", self._generate_signal)

        workflow.add_edge(START, "fetch_news")
        workflow.add_edge("fetch_news", "extract_headline")
        workflow.add_edge("extract_headline", "compute_indicators")
        workflow.add_edge("compute_indicators", "setup_probability")
        workflow.add_edge("setup_probability", "news_analysis")
        workflow.add_edge("news_analysis", "generate_signal")
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
            print(f"   ⏱️  news fetch timed out after {NEWS_FETCH_TIMEOUT_SEC}s for {state['ticker']}")
            state["news"] = []
        except Exception as e:  # keep the pipeline alive: no news just means sentiment = 0
            print(f"   ⚠️  news fetch failed for {state['ticker']}: {type(e).__name__}: {e}")
            state["news"] = []
        if not state["news"]:
            print(f"   📭 no news items for {state['ticker']}")
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
        # Uses train/setups.py + train/features.py -- the exact code the model was trained with.
        # Blocking yfinance calls run in threads so they don't stall the event loop.
        ticker = state["ticker"]
        state["benchmark"] = features.benchmark_for(ticker)
        state["setup"], state["indicators"], state["price_data"] = "", {}, None
        prices, vix, bench, earnings = await asyncio.gather(
            asyncio.to_thread(features.fetch_history, ticker, features.LIVE_PERIOD),
            asyncio.to_thread(features.fetch_vix, features.LIVE_PERIOD),
            asyncio.to_thread(features.fetch_bench, ticker, features.LIVE_PERIOD),
            asyncio.to_thread(features.fetch_earnings_dates, ticker),
        )
        # Training only saw completed sessions; use the last finished one (a half-formed bar would
        # have partial volume and a "close" that is just the last price).
        prices = features.drop_incomplete_last_bar(prices)
        table = setups.build_setup_table(prices, vix, bench, earnings)
        if table.empty:
            print(f"   ⚠️  no usable data for {ticker} (price/VIX/{state['benchmark']} missing or too short)")
            return state

        latest = table.iloc[-1]
        state["price_data"] = prices
        state["as_of"] = table.index[-1].date().isoformat()
        state["setup"] = str(latest["setup"])
        state["indicators"] = {c: float(latest[c]) for c in setups.SETUP_FEATURES}  # days_to_earn may be NaN
        state["indicators"]["Close"] = float(latest["Close"])
        return state

    async def _setup_probability(self, state: AgentState) -> AgentState:
        """P(this setup reaches its target before its stop), layered by how much evidence supports each part."""
        state["p_info"] = None
        bundle = train.get_bundle()
        if not state.get("setup") or bundle is None:
            return state
        state["p_info"] = setup_model.predict_final(bundle, state["indicators"], state["setup"])
        p = state["p_info"]
        print(f"   [{state['ticker']}] {state['setup']}: p(any day)={p['p_any']:.2f}  p(setup)={p['p_setup']:.2f}  "
              f"-> p(final)={p['p_final']:.2f}  (setup trust {p['setup_trust']:.2f}, model trust {p['model_trust']:.2f}, "
              f"validated={p['validated']})")
        return state

    _SETUP_BRIEF = {
        "pullback_buy": "a pullback inside an uptrend that has just shown a reversal candle. The trader is deciding "
                        "whether to BUY the dip: is the dip noise or an overreaction, or has something lasting damaged the business?",
        "breakout_buy": "a breakout above the 20-day high on heavy volume. The trader is deciding whether to BUY: "
                        "is there a real catalyst behind the move, or is it unexplained?",
        "rally_sell":   "a stalled bounce inside a downtrend. The trader is deciding whether to SELL: does the news give "
                        "the bounce a lasting reason, or is it just a relief rally?",
        "breakdown_sell": "a breakdown below the 20-day low on heavy volume. The trader is deciding whether to SELL: "
                          "does the news explain and confirm the weakness?",
    }

    async def _news_analysis(self, state: AgentState) -> AgentState:
        """Read the headlines in the context of the active setup. Skipped (no LLM call) when there is no
        setup or no headlines."""
        state["news_analysis"], state["sentiment_score"] = None, 0.0
        if not state.get("setup"):
            return state
        print(f"   [node] reading news for {state['ticker']} in the context of {state['setup']}...")
        prompt = f"""You are assisting a swing trader looking at {state['ticker']}.
Active setup: {self._SETUP_BRIEF[state['setup']]}

Headlines: {' | '.join(state['headlines'])}

Reply with ONLY the following JSON object, if there are no headlines provided, generatefrom the most recent top headlines of the stock that you think has the most significant, no other text:
{{"sentiment": <number from -1 (very negative for the stock) to 1 (very positive)>,
 "thesis": "improved" | "intact" | "damaged" | "unclear"   (effect of this news on the company's business outlook),
 "materiality": <number 0 to 1: how much this could lastingly change the outlook; 0 = noise>,
 "reason": "<one short sentence>"}}"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        state["news_analysis"] = self._parse_news(self._response_text(response))
        state["sentiment_score"] = state["news_analysis"]["sentiment"]
        
        return state

    @staticmethod
    def _parse_news(text: str) -> dict:
        """Parse the model's JSON; fall back to a bare number (sentiment only) if it didn't comply."""
        out = {"sentiment": 0.0, "thesis": "unclear", "materiality": 0.0, "reason": ""}
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                j = json.loads(m.group())
                out["sentiment"] = float(max(min(float(j.get("sentiment", 0.0)), 1.0), -1.0))
                out["materiality"] = float(max(min(float(j.get("materiality", 0.0)), 1.0), 0.0))
                th = str(j.get("thesis", "unclear")).lower()
                out["thesis"] = th if th in ("improved", "intact", "damaged", "unclear") else "unclear"
                out["reason"] = str(j.get("reason", ""))[:200]
                return out
            except (ValueError, TypeError):
                pass
        num = re.search(r"-?\d+(?:\.\d+)?", text)
        if num:
            out["sentiment"] = float(max(min(float(num.group()), 1.0), -1.0))
        return out

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

    async def _generate_signal(self, state: AgentState) -> AgentState:
        """BUY / SELL only when a setup is active, it has out-of-sample support, and the expected value
        after costs clears the bar (agent/decision.py). Otherwise HOLD -- with the setup, its plan and
        the reason shown if one is active."""
        print("_generate_signal")
        ind = state.get("indicators") or {}
        if not ind:  # data outage / too little history: say so, rather than "no setup"
            state["signal"], state["signal_confidence"], state["signal_score"] = "HOLD", 0.0, 0.0
            state["signal_detail"] = {"reason": "no usable data (prices, VIX or benchmark missing or too short)"}
            return state
        out = decision.decide(
            setup=state.get("setup", ""),
            p_info=state.get("p_info"),
            atr_pct=ind.get("atr_pct"),
            last_close=ind.get("Close"),
            as_of=state.get("as_of"),
            news=state.get("news_analysis"),
            params=decision.load_params(),
            days_to_earn=ind.get("days_to_earn"),
        )
        state["signal"] = out["signal"]
        state["signal_confidence"] = out["confidence"]
        state["signal_score"] = out["score"]
        state["signal_detail"] = out["detail"]
        decision.record(state, out)  # shadow log: outcomes later teach the news weight
        return state