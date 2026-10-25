"""
agent/decision.py -- BUY / SELL / HOLD for the swing setups, with every number traceable.

A signal needs a SETUP (pullback_buy / breakout_buy / rally_sell / breakdown_sell, from train/setups.py).
No setup -> HOLD. When one is active:

  p_final   probability the trade reaches its profit target before its stop within MAX_HOLD bars,
            from train/setup_model.predict_final (layered evidence, each layer weighted by how much
            out-of-sample support it has)
  news      the LLM reads the headlines IN THE CONTEXT of the setup (is the dip noise or lasting damage?
            is there a real catalyst behind the breakout?) and returns sentiment / thesis / materiality.
              - veto  : news that contradicts the trade materially (a thesis-breaking event under a dip
                        buy) blocks it. This is a risk rule, stated plainly, not a fitted weight.
              - shift : otherwise news moves p in log-odds by  gamma * (direction * sentiment * strength).
                        gamma starts from a weak prior and is learned from this agent's own logged calls
                        once their stop/target outcomes are known (refit_news).
  EV        = p * (TP/SL) + (1 - p) * avg_nonwin_R - cost_R        (in R = units of the stop distance)
  signal    = BUY/SELL when EV >= MIN_EV_R and the setup is validated; STRONG when EV >= STRONG_EV_R.

"Validated" means a setup has out-of-sample evidence behind it: either the rule beat entering on a random
day, or the model ranked outcomes within the setup (train/setup_model.py). Without that the setup is still
REPORTED (with its stop, target and time limit, as a watch-list alert) but the signal stays HOLD.
Set REQUIRE_VALIDATION = False to act on unvalidated setups -- then you are relying on rules and
history that may owe their record to hindsight selection of the tickers.

decide() is pure (no I/O) so it can be unit-tested and replayed in a backtest.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from train import features, setups

LOG_PATH = "signal_log.jsonl"
PARAMS_PATH = "decision_params.json"

# ---- decision policy: risk appetite, not predictions -----------------------------
MIN_EV_R = 0.15                   # need >= this expected R per trade after costs
STRONG_EV_R = 0.50
EARNINGS_BLACKOUT_CAL_DAYS = 14   # ~ the hold window: don't hold through earnings (relax if you want to)
REQUIRE_VALIDATION = True
VETO_MATERIALITY = 0.6            # news must be at least this material to veto a trade

# ---- prior for the news weight (log-odds shift per +1.0 of aligned news strength) --
GAMMA_PRIOR_MEAN = 0.15
GAMMA_PRIOR_SD = 0.40             # wide: we don't know; logged outcomes override this
GAMMA_MAX = 1.5

DEFAULT_PARAMS = {"news_gamma": GAMMA_PRIOR_MEAN, "news_gamma_sd": GAMMA_PRIOR_SD, "n": 0, "source": "prior"}

_BUYS = ("pullback_buy", "breakout_buy")


def _sigmoid(z): return 1.0 / (1.0 + math.exp(-z))
def _logit(p): p = min(max(p, 1e-3), 1 - 1e-3); return math.log(p / (1 - p))


# ---------------------------------------------------------------------------
# News in the context of a setup (pure)
# ---------------------------------------------------------------------------
def news_effect(setup: str, news: Optional[dict]) -> dict:
    """Turn the LLM's structured reading into (strength aligned with the trade, veto?).

    news = {"sentiment": -1..1, "thesis": "improved"|"intact"|"damaged"|"unclear", "materiality": 0..1}
    strength = direction * sentiment * (0.5 + 0.5 * materiality): positive when the news SUPPORTS the trade.
    """
    if not news:
        return {"strength": 0.0, "veto": False, "why": ""}
    d = 1.0 if setup in _BUYS else -1.0
    sent = float(np.clip(news.get("sentiment", 0.0), -1.0, 1.0))
    mat = float(np.clip(news.get("materiality", 0.0), 0.0, 1.0))
    thesis = str(news.get("thesis", "unclear"))
    against = (d > 0 and thesis == "damaged") or (d < 0 and thesis == "improved")
    veto = bool(against and mat >= VETO_MATERIALITY)
    why = f"news says the thesis is '{thesis}' (materiality {mat:.2f}) -- against this {'buy' if d > 0 else 'sell'}" if veto else ""
    return {"strength": d * sent * (0.5 + 0.5 * mat), "veto": veto, "why": why}


# ---------------------------------------------------------------------------
# The decision (pure)
# ---------------------------------------------------------------------------
def _hold(reason: str, **extra) -> dict:
    return {"signal": "HOLD", "confidence": 0.0, "score": 0.0, "detail": {"reason": reason, **extra}}


def decide(
    *,
    setup: str,
    p_info: Optional[dict],
    atr_pct: Optional[float],
    last_close: Optional[float],
    as_of=None,
    news: Optional[dict] = None,
    params: Optional[dict] = None,
    days_to_earn: Optional[float] = None,
) -> dict:
    """Return {'signal', 'confidence', 'score', 'detail'}.

    confidence = P(target before stop) after news; score = expected R per trade after costs.
    """
    params = params or DEFAULT_PARAMS
    if not setup:
        return _hold("no setup active")
    d = 1.0 if setup in _BUYS else -1.0
    if atr_pct is None or not np.isfinite(atr_pct) or atr_pct <= 0 or last_close is None:
        return _hold("no volatility estimate (indicators unavailable)", setup=setup)

    rr = setups.TP_ATR / setups.SL_ATR
    plan = {
        "entry_ref": round(float(last_close), 4),
        "stop": round(float(last_close * (1 - d * setups.SL_ATR * atr_pct / 100)), 4),
        "target": round(float(last_close * (1 + d * setups.TP_ATR * atr_pct / 100)), 4),
        "stop_pct": round(setups.SL_ATR * atr_pct, 2),
        "target_pct": round(setups.TP_ATR * atr_pct, 2),
        "max_hold_date": setups.add_trading_days(as_of, setups.MAX_HOLD + 1).date().isoformat() if as_of else None,
    }
    base = {"setup": setup, "direction": int(d), "plan": plan, "rr": rr}

    # Computed before any early exit so that vetoed / blocked setups are still LOGGED with their probability
    # and news reading: you can later check whether the veto and the blackout actually helped, and the
    # news weight is not learned only from the trades that happened to pass.
    eff = news_effect(setup, news)
    gamma = float(params["news_gamma"])
    log_fields = {}
    if p_info is not None:
        log_fields = {"p_pre_news": round(float(p_info["p_final"]), 3), "news": news,
                      "news_strength": round(eff["strength"], 3), "ev_R": None}

    if days_to_earn is not None and np.isfinite(days_to_earn) and 0 <= days_to_earn <= EARNINGS_BLACKOUT_CAL_DAYS:
        return _hold(f"earnings in {days_to_earn:.0f} day(s): inside the {setups.MAX_HOLD}-bar hold window", **base, **log_fields, watch=True)
    if p_info is None:
        return _hold("setup active but no model/evidence available", **base, watch=True)
    if eff["veto"]:
        return _hold("NEWS VETO: " + eff["why"], **base, **log_fields, watch=True)

    p_pre = float(p_info["p_final"])
    p = _sigmoid(_logit(p_pre) + gamma * eff["strength"])

    cost_r = setups.COST_PCT / (setups.SL_ATR * atr_pct)
    r_nw = float(p_info.get("r_nonwin", -1.0))
    ev = p * rr + (1 - p) * r_nw - cost_r
    validated = bool(p_info.get("validated", False)) or not REQUIRE_VALIDATION

    detail = {
        **base,
        "p_any": round(p_info["p_any"], 3), "p_setup": round(p_info["p_setup"], 3),
        "p_model": None if not np.isfinite(p_info["p_cal"]) else round(p_info["p_cal"], 3),
        "setup_trust": round(p_info["setup_trust"], 2), "model_trust": round(p_info["model_trust"], 2),
        "p_pre_news": round(p_pre, 3), "p_final": round(p, 3),
        "news": news, "news_strength": round(eff["strength"], 3),
        "gamma": round(gamma, 3),
        "gamma_source": params.get("source", "prior") + (f" (n={params['n']})" if params.get("n") else ""),
        "r_nonwin": round(r_nw, 3), "cost_R": round(cost_r, 3), "ev_R": round(ev, 3),
        "validated": validated, "watch": False,
    }

    if ev >= MIN_EV_R and validated:
        strong = ev >= STRONG_EV_R
        label = ("STRONG " if strong else "") + ("BUY" if d > 0 else "SELL")
        return {"signal": label, "confidence": round(p, 3), "score": round(ev, 3), "detail": detail}

    why = ("setup active but not validated out-of-sample (evidence too thin to act on)" if not validated
           else f"expected value {ev:+.2f}R is below the {MIN_EV_R}R bar")
    detail.update(reason=why, watch=True)
    return {"signal": "HOLD", "confidence": round(p, 3), "score": round(ev, 3), "detail": detail}


# ---------------------------------------------------------------------------
# Parameters + shadow log
# ---------------------------------------------------------------------------
def load_params(path: str = PARAMS_PATH) -> dict:
    params = dict(DEFAULT_PARAMS)
    try:
        with open(path) as f:
            params.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return params


def save_params(params: dict, path: str = PARAMS_PATH) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def record(state: dict, out: dict, path: str = LOG_PATH) -> None:
    """Append a decision to the shadow log (only when a setup was active and had a probability).
    Never raises: logging must not break a run. Outcomes are labelled later from price history,
    so even calls that were HOLD teach the news weight."""
    d = out.get("detail", {})
    if "p_pre_news" not in d:
        return
    try:
        news = d.get("news") or {}
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": state.get("ticker"), "as_of": state.get("as_of"),
            "setup": d["setup"], "direction": d["direction"], "atr_pct": state.get("indicators", {}).get("atr_pct"),
            "p_pre_news": d["p_pre_news"], "news_strength": d["news_strength"],
            "sentiment": news.get("sentiment"), "thesis": news.get("thesis"), "materiality": news.get("materiality"),
            "headline": (state.get("headlines") or [""])[0][:160],
            "signal": out["signal"], "ev_R": d.get("ev_R"), "reason": d.get("reason", "")[:120],
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print(f"   ⚠️  could not write decision log: {e}")


def read_log(path: str = LOG_PATH) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Learning the news weight from the agent's own matured calls
# ---------------------------------------------------------------------------
def label_outcomes(rows: list[dict], fetch: Callable[[str], pd.DataFrame]) -> list[dict]:
    """Attach the stop/target outcome (y) of each logged setup, using the SAME barrier rules as training.

    Skipped: rows whose window isn't complete, and rows that overlap an already-kept row for the same
    ticker and direction (consecutive days share most of their outcome window).
    """
    uniq: dict[tuple, dict] = {}
    for r in rows:  # newest log entry wins for the same ticker/day/setup
        uniq[(r.get("ticker"), r.get("as_of"), r.get("setup"))] = r
    ordered = sorted(uniq.values(), key=lambda r: (str(r.get("ticker")), str(r.get("as_of"))))

    ohlc: dict[str, Optional[pd.DataFrame]] = {}
    last_kept: dict[tuple, int] = {}
    out: list[dict] = []
    for r in ordered:
        tkr = r.get("ticker")
        if tkr not in ohlc:
            try:
                df = fetch(tkr)
                ohlc[tkr] = df if df is not None and not df.empty else None
            except Exception as e:
                print(f"   ⚠️  could not fetch prices for {tkr}: {e}")
                ohlc[tkr] = None
        df = ohlc[tkr]
        atr_pct = r.get("atr_pct")
        if df is None or not r.get("as_of") or atr_pct is None:
            continue
        i = int(df.index.get_indexer([pd.Timestamp(r["as_of"])])[0])
        if i < 0:
            continue
        key = (tkr, r.get("direction"))
        if key in last_kept and i - last_kept[key] < setups.MAX_HOLD:
            continue
        res = setups.barrier_outcome(df["Open"].to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float),
                                     df["Close"].to_numpy(float), i, float(r["direction"]), float(atr_pct))
        if res is None:
            continue  # outcome hasn't played out yet
        last_kept[key] = i
        out.append({**r, "y": res[0], "R": res[1]})
    return out


def posterior_gamma(x: np.ndarray, y: np.ndarray, offset: np.ndarray, prior_mean: float, prior_sd: float) -> tuple[float, float]:
    """Bayesian logistic regression with a fixed offset: P(y) = sigmoid(offset + gamma * x),
    gamma ~ N(prior_mean, prior_sd^2). Newton's method. Returns (posterior mean, posterior sd).
    With few outcomes it stays at the prior; with many it follows the data."""
    g = float(prior_mean)
    for _ in range(50):
        q = 1.0 / (1.0 + np.exp(-(offset + g * x)))
        grad = float(np.sum(x * (y - q))) - (g - prior_mean) / prior_sd ** 2
        hess = -float(np.sum(x * x * q * (1 - q))) - 1.0 / prior_sd ** 2
        step = grad / hess
        g -= step
        if abs(step) < 1e-9:
            break
    return g, math.sqrt(1.0 / -hess)


def refit_news(log_path: str = LOG_PATH, params_path: str = PARAMS_PATH,
               fetch: Optional[Callable[[str], pd.DataFrame]] = None) -> dict:
    """Update the news weight from logged setups whose outcomes are now known. Safe to call anytime."""
    fetch = fetch or (lambda t: features.fetch_history(t, "2y"))
    params = load_params(params_path)
    try:
        labeled = label_outcomes(read_log(log_path), fetch)
    except Exception as e:
        print(f"   ⚠️  news refit skipped: {type(e).__name__}: {e}")
        return params
    labeled = [r for r in labeled if r.get("p_pre_news") and r.get("news_strength") is not None]
    if not labeled:
        return params
    x = np.array([r["news_strength"] for r in labeled], float)
    y = np.array([r["y"] for r in labeled], float)
    off = np.array([_logit(r["p_pre_news"]) for r in labeled], float)
    mean, sd = posterior_gamma(x, y, off, GAMMA_PRIOR_MEAN, GAMMA_PRIOR_SD)
    params.update(news_gamma=float(np.clip(mean, 0.0, GAMMA_MAX)),  # never invert the LLM's sign
                  news_gamma_sd=float(sd), n=len(labeled), source="learned",
                  updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    save_params(params, params_path)
    return params