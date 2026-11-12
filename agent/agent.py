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

import logging

import yfinance as yf
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict, Optional, Any

from agent import decision, newsfeed
from train import features, setup_model, setups, train
from config import get_config

cfg = get_config()


class AgentState(TypedDict):
    ticker: str
    trigger: str                 # "news" (headline changed) or "scan" (daily technical scan)
    news: list[dict]
    headlines: list[str]
    headline_relevance: dict    # early LLM gate: {"relevant": bool, "reason": str}
    price_data: Optional[pd.DataFrame]
    indicators: dict            # setup features of the last COMPLETED session (+ Close)
    as_of: str                  # that session's date
    benchmark: str              # reference for the relative-strength features (e.g. SPY)
    setup: str                  # best-matching setup: pullback_buy | breakout_buy | rally_sell | breakdown_sell
    setup_match: float          # how fully formed it is, 0-1 (1 = every condition of the strict rule is met)
    setup_hard: bool            # True when the strict rule fired
    setup_gaps: list            # what the setup is still missing (empty when fully formed)
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
        logging.info(f"Initializing TickerAgent for model {model}...")
        api_key = os.getenv("API_KEY")
        provider = os.getenv("LLM_PROVIDER")
        logging.info(f"   Provider: {provider}")
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

        logging.info(f"   init_chat_model kwargs: { {k: v for k, v in kwargs.items() if 'key' not in k} }")
        self.llm = init_chat_model(**kwargs)

        workflow = StateGraph(state_schema=AgentState)

        workflow.add_node("fetch_news", self._fetch_news)
        workflow.add_node("extract_headline", self._extract_headline)
        workflow.add_node("check_headline_relevance", self._check_headline_relevance)
        workflow.add_node("skip_irrelevant", self._skip_irrelevant)
        workflow.add_node("compute_indicators", self._compute_indicators)
        workflow.add_node("setup_probability", self._setup_probability)
        workflow.add_node("news_analysis", self._news_analysis)
        workflow.add_node("generate_signal", self._generate_signal)

        workflow.add_edge(START, "fetch_news")
        workflow.add_edge("fetch_news", "extract_headline")
        workflow.add_edge("extract_headline", "check_headline_relevance")
        workflow.add_conditional_edges(
            "check_headline_relevance",
            self._route_after_headline_relevance,
            {"relevant": "compute_indicators", "irrelevant": "skip_irrelevant"},
        )
        workflow.add_edge("compute_indicators", "setup_probability")
        workflow.add_edge("setup_probability", "news_analysis")
        workflow.add_edge("news_analysis", "generate_signal")
        workflow.add_edge("generate_signal", END)
        workflow.add_edge("skip_irrelevant", END)

        self.workflow = workflow
        self.app = self.workflow.compile()
        logging.debug(f"TickerAgent workflow compiled.")

    async def run(self, ticker: str, trigger: str = "news") -> AgentState:
        logging.info(f"Running agent for {ticker} ({trigger})...")
        initial_state: AgentState = {"ticker": ticker, "trigger": trigger}
        try:
            result = await self.app.ainvoke(initial_state)
            logging.info(f"Finished agent for {ticker}.")
            return result
        except Exception as e:
            logging.error(f"Error in TickerAgent.run for {ticker}: {e}")
            raise

    async def _fetch_news(self, state: AgentState) -> AgentState:
        logging.info(f"  [node] fetching news for {state['ticker']}")
        try:
            state["news"] = await asyncio.wait_for(
                asyncio.to_thread(newsfeed.fetch_news, state["ticker"], self.newsLimit),
                timeout=cfg.runtime.news_fetch_timeout_sec

            )
        except asyncio.TimeoutError:
            logging.error(f"   news fetch timed out after {cfg.runtime.news_fetch_timeout_sec}s for {state['ticker']}")
            state["news"] = []
        except Exception as e:  # keep the pipeline alive: no news just means sentiment = 0
            logging.error(f"   news fetch failed for {state['ticker']}: {type(e).__name__}: {e}")
            state["news"] = []
        if not state["news"]:
            logging.warn(f"   no news items for {state['ticker']}")
        return state

    async def _extract_headline(self, state: AgentState) -> AgentState:
        logging.info(f"   [node] extracting headline for {state['ticker']}...")
        state["headlines"] = []
        for item in state.get("news") or []:
            n = newsfeed.normalize(item)
            text = n["summary"] or n["title"]
            if text:
                state["headlines"].append(text)
        return state

    async def _check_headline_relevance(self, state: AgentState) -> AgentState:
        """Early LLM gate: stop before the expensive analysis when the news cannot move this ticker.

        Daily technical scans are never gated: they must still run when the latest headline is
        unrelated (or there is no news at all).
        """
        state["headline_relevance"] = {"relevant": True, "reason": ""}
        if state.get("trigger") == "scan":
            return state

        headlines = state.get("headlines") or []
        if not headlines:
            return state

        logging.debug(
            f"   [node] checking whether the headline is price-relevant for {state['ticker']}..."
        )
        headline_text = "\n".join(f"- {h}" for h in headlines)
        prompt = f"""
You are screening news for a swing-trading screener.
Ticker: {state['ticker']}

Headlines:
{headline_text}

State the relevancy if this news could plausibly affect {state['ticker']}'s price
Reply with ONLY this JSON object:
{{"relevant": <true or false>, "reason": "<one short sentence>"}}
        """
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        except Exception as e:
            logging.error(
                f"   headline relevance check failed for {state['ticker']}: {type(e).__name__}: {e}"
            )
            return state  # fail open: a broken gate must not suppress the setup analysis

        state["headline_relevance"] = self._parse_relevance(self._response_text(response))
        rel = state["headline_relevance"]
        logging.info(f"   price relevance: {rel['relevant']} -- {rel['reason']}")
        return state

    @staticmethod
    def _parse_relevance(text: str) -> dict:
        """Parse the model's relevance JSON; fail open (relevant=True) on malformed output."""
        out = {"relevant": True, "reason": ""}
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return out
        try:
            payload = json.loads(match.group())
        except (ValueError, TypeError):
            return out
        if not isinstance(payload, dict):
            return out

        value = payload.get("relevant", True)
        if value is None:
            out["relevant"] = True
        elif isinstance(value, str):
            out["relevant"] = value.strip().lower() in ("true", "yes", "1")
        else:
            out["relevant"] = bool(value)
        out["reason"] = str(payload.get("reason", ""))[:200]
        return out

    @staticmethod
    def _route_after_headline_relevance(state: AgentState) -> str:
        relevance = state.get("headline_relevance") or {}
        return "irrelevant" if relevance.get("relevant") is False else "relevant"

    async def _skip_irrelevant(self, state: AgentState) -> AgentState:
        """Short-circuit an unrelated headline: report HOLD without fetching indicators or calling the LLM again."""
        relevance = state.get("headline_relevance") or {}
        reason = relevance.get("reason") or "news is not price-relevant"
        logging.info(f"   [{state['ticker']}] skipping pipeline: {reason}")
        state["signal"] = "HOLD"
        state["signal_confidence"] = 0.0
        state["signal_score"] = 0.0
        state["signal_detail"] = {
            "reason": f"headline not price-relevant for {state['ticker']}: {reason}",
            "headline_relevance": relevance,
        }
        return state

    async def _compute_indicators(self, state: AgentState) -> AgentState:
        # Uses train/setups.py + train/features.py -- the exact code the model was trained with.
        # Blocking yfinance calls run in threads so they don't stall the event loop.
        ticker = state["ticker"]
        state["benchmark"] = features.benchmark_for(ticker)
        state["setup"], state["indicators"], state["price_data"] = "", {}, None
        state["setup_match"], state["setup_hard"], state["setup_gaps"] = 0.0, False, []
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
        state["setup"] = str(latest["setup"])            # always one of the four: the best match
        state["setup_match"] = float(latest["match"])
        state["setup_hard"] = bool(latest["hard"] == 1)
        state["setup_gaps"] = setups.describe_gaps(latest, state["setup"])
        state["indicators"] = {c: float(latest[c]) for c in setups.SETUP_FEATURES}  # days_to_earn may be NaN
        state["indicators"]["Close"] = float(latest["Close"])
        return state

    async def _setup_probability(self, state: AgentState) -> AgentState:
        """P(this setup reaches its target before its stop), layered by how much evidence supports each part."""
        state["p_info"] = None
        bundle = train.get_bundle()
        if not state.get("setup") or bundle is None:
            return state
        state["p_info"] = setup_model.predict_final(bundle, state["indicators"], state["setup"], state["setup_hard"])
        p = state["p_info"]
        logging.info(f"   [{state['ticker']}] {state['setup']} match {state['setup_match']:.2f} ({p['bucket']}): p(any day)={p['p_any']:.2f}  p(this kind of day)={p['p_group']:.2f}  -> p(final)={p['p_final']:.2f} (evidence {p['setup_trust']:.2f}, model {p['model_trust']:.2f}, validated={p['validated']})")
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
        """Read the headlines in the context of the best-matching setup, however fully formed it is.
        Skipped (no LLM call) only when there are no headlines to read, or no data at all."""
        state["news_analysis"], state["sentiment_score"] = None, 0.0
        if not state.get("setup") or not state.get("headlines"):
            return state
        logging.info(f"   [node] reading news for {state['ticker']} in the context of {state['setup']}...")
        formed = ("fully formed" if state.get("setup_hard") else
                  f"{int(state.get('setup_match', 0) * 100)}% of the way there; still missing: " + ("; ".join(state.get("setup_gaps") or ["nothing specific"])))
        
        prompt = f"""
You are assisting a swing trader looking at {state['ticker']}.
Closest setup: {self._SETUP_BRIEF[state['setup']]}
Chart status: the pattern is {formed}.

Headlines: {' | '.join(state['headlines'])}

Reply with ONLY the following JSON object, if there are no headlines provided, generatefrom the most recent top headlines of the stock that you think has the most significant, no other text:
{{"sentiment": <number from -1 (very negative for the stock) to 1 (very positive)>,
 "thesis": "improved" | "intact" | "damaged" | "unclear"   (effect of this news on the company's business outlook),
 "materiality": <number 0 to 1: how much this could lastingly change the outlook; 0 = noise>,
 "reason": "<one short sentence>"}}
        """
        
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
        """BUY / SELL only when the evidence for this kind of day (setup + how fully formed) holds out of
        sample and the expected value after costs clears the bar (agent/decision.py). Otherwise HOLD, with
        the setup, how formed it is, and the reason."""
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
            match=state.get("setup_match", 1.0),
            hard=state.get("setup_hard", True),
            gaps=state.get("setup_gaps"),
        )
        
        state["signal"] = out["signal"]
        state["signal_confidence"] = out["confidence"]
        state["signal_score"] = out["score"]
        state["signal_detail"] = out["detail"]
        decision.record(state, out)  # shadow log: outcomes later teach the news weight
        return state