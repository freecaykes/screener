"""
agent/decision.py -- turn the agent's evidence into a signal WITHOUT hand-picked weights.

The old formula added `delta*0.5 + sentiment*0.3 + pullback*0.15 - (vix-18)*0.012 + ...`:
different units, weights nobody measured, and pullback/RSI/VIX counted twice (they are
already XGBoost inputs). This module replaces it with an expected-value decision:

    E[move]  =  beta * xgboost_delta          # how much of the model's forecast is real
              + gamma * sentiment * sigma     # how much LLM sentiment moves price, in sigma units
    sigma    =  excess_vol_20d * sqrt(horizon)  # volatility of the EXCESS move over the horizon
    P(win)   =  Phi( (|E[move]| - costs) / sigma )   # chance the move beats costs, as a probability

All moves here are EXCESS moves: the stock minus its benchmark (SPY by default). A BUY is
"expected to outperform the benchmark", so sigma is the volatility of that difference, not of
the stock alone. To capture it you would pair the stock with an opposite benchmark position;
trading only the stock leg also exposes you to the market's direction.
  * beta   : MEASURED. Out-of-sample calibration slope x trust weight from the walk-forward
             validation (train/timing.py). Noise model -> beta ~ 0 -> forecast ~ 0 -> HOLD.
  * gamma  : LEARNED. The LLM sentiment has no price history to train on, so it starts from
             a weak prior and is updated (Bayesian shrinkage) from the agent's own logged
             calls once their outcomes have matured. See refit_sentiment().
  * pullback / RSI / VIX : no hand weight. They are model inputs, so their effect is already
             inside the model's forecast; volatility enters through sigma.
  * thresholds : probabilities (P_ACT, P_STRONG) and a cost assumption. Those are risk
             appetite and trading costs, not predictions -- tune them on a backtest.
  * confidence : the actual P(win), not an invented number.

decide() is pure (no I/O) so it can be unit-tested and replayed in a backtest.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from train import features, timing

LOG_PATH = "signal_log.jsonl"
PARAMS_PATH = "decision_params.json"

# ---- decision policy: risk appetite and costs (not predictions) -------------
COST_PCT = 0.10        # assumed round-trip cost + slippage, in % of price (a hedged pair costs ~2x)
EARNINGS_BLACKOUT_CAL_DAYS = 4   # abstain if earnings are due within this many calendar days
P_ACT = 0.55           # need >= 55% chance the move beats costs to say BUY / SELL
P_STRONG = 0.65        # ... and >= 65% for STRONG BUY / STRONG SELL
MIN_SIGMA_PCT = 0.25   # numerical floor so a dead-calm stretch can't divide by ~0

# ---- prior for LLM sentiment sensitivity (units: sigma of the horizon move per +1.0 sentiment) ----
GAMMA_PRIOR_MEAN = 0.05
GAMMA_PRIOR_SD = 0.15  # wide: we genuinely don't know; data overrides this as it accumulates
GAMMA_MAX = 0.5

DEFAULT_PARAMS = {
    "sentiment_gamma": GAMMA_PRIOR_MEAN,
    "sentiment_gamma_sd": GAMMA_PRIOR_SD,
    "n": 0,
    "source": "prior",
}


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# The decision (pure)
# ---------------------------------------------------------------------------
def _hold(reason: str) -> dict:
    return {"signal": "HOLD", "confidence": 0.0, "score": 0.0, "detail": {"reason": reason}}


def decide(
    *,
    pred_delta: float,
    base_metrics: Optional[dict],
    daily_vol_pct: Optional[float],
    sentiment: float,
    params: Optional[dict] = None,
    horizon: int = timing.BASE_HORIZON,
    as_of=None,
    manifest: Optional[dict] = None,
    days_to_earn: Optional[float] = None,
) -> dict:
    """Return {'signal', 'confidence', 'score', 'detail'}.

    pred_delta    raw XGBoost expected EXCESS % move (vs benchmark) over `horizon` trading days
    base_metrics  validation metrics for that horizon (needs calib_beta, trust_weight)
    daily_vol_pct 20-day std of the daily EXCESS return, in %  (features: excess_vol_20d)
    sentiment     LLM score in [-1, 1]
    days_to_earn  calendar days to the next earnings date, if known
    """
    params = params or DEFAULT_PARAMS
    if daily_vol_pct is None or not np.isfinite(daily_vol_pct):
        return _hold("no volatility estimate (indicators unavailable)")
    if days_to_earn is not None and np.isfinite(days_to_earn) and 0 <= days_to_earn <= EARNINGS_BLACKOUT_CAL_DAYS:
        # A binary earnings gap dwarfs a 3-day statistical edge, so don't trade through it.
        return _hold(f"earnings in {days_to_earn:.0f} day(s): abstaining through the announcement")

    sigma = max(float(daily_vol_pct) * math.sqrt(horizon), MIN_SIGMA_PCT)

    m = base_metrics or {}
    beta_raw = float(m.get("calib_beta", 0.0))
    trust = float(m.get("trust_weight", 0.0))
    beta = beta_raw * trust
    gamma = float(params["sentiment_gamma"])

    mu_model = beta * float(pred_delta)
    mu_sent = gamma * float(sentiment) * sigma
    mu = mu_model + mu_sent

    # Probability that a trade in each direction clears costs, assuming the realised move
    # ~ Normal(mu, sigma). For a long: P(R > cost) = Phi((mu - cost) / sigma).
    p_buy = _phi((mu - COST_PCT) / sigma)
    p_sell = _phi((-mu - COST_PCT) / sigma)

    if p_buy >= P_STRONG:
        signal, confidence = "STRONG BUY", p_buy
    elif p_buy >= P_ACT:
        signal, confidence = "BUY", p_buy
    elif p_sell >= P_STRONG:
        signal, confidence = "STRONG SELL", p_sell
    elif p_sell >= P_ACT:
        signal, confidence = "SELL", p_sell
    else:
        signal, confidence = "HOLD", 1.0 - max(p_buy, p_sell)  # chance neither side clears costs

    # When to look again: the timing manifest's 80%-of-move day if it agrees with the call
    # and passed validation, else the base horizon.
    target_date = None
    if signal != "HOLD" and as_of is not None:
        want = "up" if "BUY" in signal else "down"
        h = horizon
        if manifest and manifest.get("reliable") and manifest.get("direction") == want:
            h = int(manifest["reach_horizon"])
        target_date = timing.add_trading_days(as_of, h).date().isoformat()

    return {
        "signal": signal,
        "confidence": round(float(confidence), 3),
        "score": round(mu / sigma, 3),  # expected move in sigma units (signed)
        "detail": {
            "horizon": horizon,
            "mu_model_pct": round(mu_model, 4),
            "mu_sentiment_pct": round(mu_sent, 4),
            "mu_total_pct": round(mu, 4),
            "sigma_pct": round(sigma, 4),
            "p_win": round(float(max(p_buy, p_sell)), 3),
            "cost_pct": COST_PCT,
            "beta_raw": round(beta_raw, 3),
            "trust_weight": round(trust, 3),
            "beta_eff": round(beta, 3),
            "gamma": round(gamma, 4),
            "gamma_source": params.get("source", "prior")
            + (f" (n={params.get('n', 0)})" if params.get("n") else ""),
            "target_date": target_date,
        },
    }


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
    """Append one decision to the shadow log. Never raises (logging must not break a run)."""
    d = out.get("detail", {})
    if "sigma_pct" not in d:  # the no-data HOLD has nothing worth learning from
        return
    try:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": state.get("ticker"),
            "as_of": state.get("as_of"),
            "horizon": d["horizon"],
            "benchmark": state.get("benchmark"),  # marks the row as market-relative
            "headline": (state.get("headlines") or [""])[0][:160],
            "sentiment": float(state.get("sentiment_score", 0.0)),
            "pred_delta": state.get("predicted_delta"),
            "mu_model_pct": d["mu_model_pct"],
            "sigma_pct": d["sigma_pct"],
            "signal": out["signal"],
            "p_win": d["p_win"],
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print(f"   could not write decision log: {e}")


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
# Learning the sentiment weight from the agent's own matured calls
# ---------------------------------------------------------------------------
def label_outcomes(rows: list[dict], fetch: Callable[[str], pd.DataFrame]) -> list[dict]:
    """Attach the realised EXCESS % move (stock minus benchmark) over each row's horizon.

    Measured from the as_of close. Skipped: rows whose horizon hasn't elapsed yet, and
    legacy rows logged before the market-relative target (no "benchmark" field -- their
    forecast was on a different basis). Rows are also thinned so a ticker's kept rows are
    at least `horizon` trading days apart, otherwise consecutive days share most of their
    outcome window and the sample looks bigger than it is.
    """
    # newest log entry wins for identical (ticker, as_of, headline)
    uniq: dict[tuple, dict] = {}
    for r in rows:
        uniq[(r.get("ticker"), r.get("as_of"), r.get("headline"))] = r
    ordered = sorted(uniq.values(), key=lambda r: (str(r.get("ticker")), str(r.get("as_of"))))

    closes: dict[str, pd.Series] = {}

    def close_of(sym: str) -> pd.Series:
        if sym not in closes:
            try:
                df = fetch(sym)
                closes[sym] = df["Close"] if df is not None and not df.empty else pd.Series(dtype=float)
            except Exception as e:
                print(f"   could not fetch prices for {sym}: {e}")
                closes[sym] = pd.Series(dtype=float)
        return closes[sym]

    last_kept: dict[str, int] = {}
    out: list[dict] = []
    for r in ordered:
        tkr, bsym, h = r.get("ticker"), r.get("benchmark"), int(r.get("horizon", timing.BASE_HORIZON))
        if not bsym or not r.get("as_of"):
            continue
        c = close_of(tkr)
        bc = close_of(bsym)
        if c.empty or bc.empty:
            continue
        bc = bc.reindex(c.index, method="ffill")
        i = int(c.index.get_indexer([pd.Timestamp(r["as_of"])])[0])
        if i < 0 or i + h >= len(c) or pd.isna(bc.iloc[i]) or pd.isna(bc.iloc[i + h]):
            continue  # unknown date, or the outcome hasn't happened yet
        if tkr in last_kept and i - last_kept[tkr] < h:
            continue  # overlaps the previously kept row
        last_kept[tkr] = i
        stock_ret = c.iloc[i + h] / c.iloc[i] - 1.0
        bench_ret = bc.iloc[i + h] / bc.iloc[i] - 1.0
        out.append({**r, "ret_pct": float((stock_ret - bench_ret) * 100.0)})
    return out


def posterior_gamma(x: np.ndarray, y: np.ndarray, prior_mean: float, prior_sd: float) -> tuple[float, float]:
    """Bayesian linear regression through the origin, y = gamma * x + noise.

    Gaussian prior N(prior_mean, prior_sd^2). Returns (posterior mean, posterior sd).
    With few observations this stays at the prior; with many it follows the data.
    """
    if len(x) == 0:
        return prior_mean, prior_sd
    noise_var = max(float(np.var(y)), 0.25)  # y is in sigma units, so ~1 for a well-scaled model
    precision = 1.0 / prior_sd**2 + float(np.dot(x, x)) / noise_var
    mean = (prior_mean / prior_sd**2 + float(np.dot(x, y)) / noise_var) / precision
    return mean, math.sqrt(1.0 / precision)


def refit_sentiment(
    log_path: str = LOG_PATH,
    params_path: str = PARAMS_PATH,
    fetch: Optional[Callable[[str], pd.DataFrame]] = None,
) -> dict:
    """Update the sentiment weight from matured log rows and save it. Safe to call anytime."""
    fetch = fetch or (lambda t: features.fetch_history(t, "2y"))
    params = load_params(params_path)
    try:
        labeled = label_outcomes(read_log(log_path), fetch)
    except Exception as e:  # network trouble etc.: keep whatever we had
        print(f"    sentiment refit skipped: {type(e).__name__}: {e}")
        return params
    if not labeled:
        return params

    x = np.array([r["sentiment"] for r in labeled], dtype=float)
    # What the model forecast did NOT explain, in sigma units -- that is what sentiment must explain.
    y = np.array([(r["ret_pct"] - r["mu_model_pct"]) / r["sigma_pct"] for r in labeled], dtype=float)
    mean, sd = posterior_gamma(x, y, GAMMA_PRIOR_MEAN, GAMMA_PRIOR_SD)

    params.update(
        sentiment_gamma=float(np.clip(mean, 0.0, GAMMA_MAX)),  # never invert the LLM's sign
        sentiment_gamma_sd=float(sd),
        n=len(labeled),
        source="learned",
        updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    save_params(params, params_path)
    return params