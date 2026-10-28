"""
train/setup_model.py -- learns WHEN a setup works, and says honestly whether it can.

For every day a setup fires on one of YOUR tickers we know what happened next (did price reach the
profit target before the stop within MAX_HOLD bars? -- see setups.barrier_outcome). The model
predicts that probability from the direction-normalised setup features.

Evidence, not hope:
  * Validation is walk-forward BY DATE with a purge (a setup's outcome looks MAX_HOLD bars ahead, so
    training rows too close to a test block are removed). Tickers share dates (market days), so rows
    are never split across train/test by ticker.
  * Skill = fold AUC computed WITHIN each setup type (does it rank winners above losers among setups of the
    same kind?), so learning that shorts and longs have different base win rates earns nothing. The gate needs both a meaningful edge over
    0.5 and a t-statistic across folds; fold-to-fold spread already reflects overlapping outcomes.
  * The model's probabilities are recalibrated (Platt) on the out-of-sample predictions and shrunk
    toward each setup's historical win rate by a soft trust weight, so a weak model can't swing a
    decision on its own.

With few tickers there are few setups (a ticker gives roughly 5-30 per year across all four types),
so expect the gate to reject often. That is the correct answer when there isn't enough evidence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import logging

from train import features, setups
from train.setups import SETUP_FEATURES

# ---- gate (tune these) --------------------------------------------------------
MIN_IC_EDGE = 0.03       # mean out-of-sample rank correlation (prediction vs realised R, within setup) at least this
MIN_T_STAT = 2.5         # ... and mean(IC) / stderr across folds must reach this
MIN_FOLDS = 5            # ... with at least this many scorable folds
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 12
MIN_ROWS_TO_FIT = 150    # below this many labelled setups there is nothing to learn from
BASE_SHRINK_N = 30       # a setup type with fewer rows than this leans on the pooled win rate

# Rule-level gate: does a setup beat entering on a RANDOM day (same direction, same barriers)?
MIN_RULE_N = 30          # labelled occurrences of the setup
MIN_RULE_MONTHS = 12     # distinct calendar months containing the setup
MIN_LIFT_R = 0.10        # net-of-cost lift over the any-day baseline, in R (stop distances)
RULE_MIN_T = 2.0         # month-clustered t-stat of that lift


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _ConstProb:
    """Stand-in when the training labels are all one class (a classifier can't be fit)."""
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def make_classifier() -> XGBClassifier:
    # Small and heavily regularised: a few hundred overlapping, noisy outcomes.
    return XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )


def fit_classifier(X: pd.DataFrame, y: pd.Series):
    if pd.Series(y).nunique() < 2:
        return _ConstProb(float(np.mean(y)))
    return make_classifier().fit(X, y)


def predict_p(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset(tickers: list[str], period: Optional[str] = None):
    """Pooled labelled setups for YOUR tickers only, plus the any-day control group.

    Returns (D, A). D has one row per setup occurrence: date, ticker, setup, SETUP_FEATURES,
    y (target-before-stop), R (result in stop-distance units), bars. A has one row per (ticker, day,
    direction): what entering on ANY day with the same barriers would have done -- the thing a setup
    has to beat (quality stocks drift up, so a long can win with no timing skill at all).
    """
    period = period or "10y"
    vix = features.fetch_vix(period)
    if vix.empty:
        raise RuntimeError("Could not download ^VIX; refusing to train without it.")
    benches: dict[str, pd.Series] = {}
    for sym in sorted({features.benchmark_for(t) for t in tickers}):
        hist = features.fetch_history(sym, period)
        if hist.empty:
            raise RuntimeError(f"Could not download benchmark {sym}.")
        benches[sym] = hist["Close"]

    frames, any_frames = [], []
    for tkr in tickers:
        try:
            prices = features.fetch_history(tkr, period)
            earnings = features.fetch_earnings_dates(tkr)
            table = setups.build_setup_table(prices, vix, benches[features.benchmark_for(tkr)], earnings)
        except Exception as e:
            logging.info(f"  ⚠️  {tkr}: skipped ({e})")
            continue
        if table.empty:
            logging.info(f"  ⚠️  {tkr}: no usable rows, skipped")
            continue

        for d in (1.0, -1.0):  # what entering on a random day would have done, same barriers
            lab_a = setups.label_rows(prices, table.assign(direction=d))
            lab_a = lab_a[["direction", "atr_pct", "y", "R"]].copy()
            lab_a["ticker"], lab_a["date"] = tkr, lab_a.index
            any_frames.append(lab_a.reset_index(drop=True))

        lab = setups.label_rows(prices, table[table["setup"] != ""])
        lab["ticker"], lab["date"] = tkr, lab.index
        frames.append(lab.reset_index(drop=True))
        counts = lab["setup"].value_counts().to_dict()
        logging.info(f"  → {tkr}: {len(lab)} labelled setups {counts}")

    if not frames:
        raise ValueError("No setups found for the given tickers")
    return pd.concat(frames, ignore_index=True), pd.concat(any_frames, ignore_index=True)


def baselines_from(A: pd.DataFrame) -> dict:
    """Any-day win rate (target before stop) per direction."""
    return {"long": float(A.loc[A["direction"] > 0, "y"].mean()), "short": float(A.loc[A["direction"] < 0, "y"].mean())}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _auc(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()  # average ranks handle ties
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_within_groups(p, y, groups) -> float:
    """AUC computed WITHIN each group and combined by number of positive/negative pairs.

    Pooled AUC would reward the model for learning that setup types have different base win rates
    (e.g. shorts vs longs). The gate is about picking better instances within a setup, so only
    comparisons inside the same setup type count.
    """
    p, y, groups = np.asarray(p, float), np.asarray(y, int), np.asarray(groups)
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        n1 = int(y[m].sum()); n0 = int(m.sum()) - n1
        if n1 == 0 or n0 == 0:
            continue
        num += _auc(p[m], y[m]) * n1 * n0
        den += n1 * n0
    return float(num / den) if den > 0 else float("nan")


def _ic_within_groups(p, r, groups) -> float:
    """Spearman rank correlation between predicted probability and realised R, within each setup
    type, combined by group size. Ranking the money result (R) -- not just win/loss -- is what
    matters: wide barriers (high ATR) lower the win rate by causing time-outs, which an AUC on wins
    rewards even though nothing is earned by knowing it."""
    p, r, groups = np.asarray(p, float), np.asarray(r, float), np.asarray(groups)
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 8:
            continue
        c = pd.Series(p[m]).corr(pd.Series(r[m]), method="spearman")
        if pd.notna(c):
            num += float(c) * int(m.sum()); den += int(m.sum())
    return float(num / den) if den > 0 else float("nan")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(p, y, ridge: float = 1.0) -> tuple[float, float]:
    """Recalibrate probabilities: P(y) = sigmoid(a + b * logit(p)). b ~ 1 means well calibrated,
    b ~ 0 means the model's ranking carries no information. b is nudged toward 1 (weakly) and clipped."""
    y = np.asarray(y, float)
    z = _logit(p)
    a, b = float(_logit(y.mean())), 0.0
    for _ in range(30):
        q = _sigmoid(a + b * z); w = q * (1 - q)
        g = np.array([np.sum(y - q), np.sum((y - q) * z) - ridge * (b - 1.0)])
        H = np.array([[np.sum(w), np.sum(w * z)], [np.sum(w * z), np.sum(w * z * z) + ridge]])
        step = np.linalg.solve(H, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    b = float(np.clip(b, 0.0, 1.5))
    for _ in range(10):  # re-fit the intercept given the (possibly clipped) slope
        q = _sigmoid(a + b * z)
        a += float(np.sum(y - q) / max(np.sum(q * (1 - q)), 1e-9))
    return float(a), b


def walk_forward_validate(D: pd.DataFrame, n_folds: int = 8, first_test_frac: float = 0.4) -> dict:
    D = D.sort_values("date").reset_index(drop=True)
    dates = np.sort(D["date"].unique())
    n = len(dates)
    edges = np.linspace(int(n * first_test_frac), n, n_folds + 1).astype(int)
    purge = pd.offsets.BDay(setups.MAX_HOLD + 2)

    ics, aucs, skills, oos = [], [], [], []
    for k in range(n_folds):
        lo, hi = int(edges[k]), int(edges[k + 1])
        if hi <= lo:
            continue
        t0, t1 = pd.Timestamp(dates[lo]), pd.Timestamp(dates[hi - 1])
        tr = D[D["date"] < t0 - purge]
        te = D[(D["date"] >= t0) & (D["date"] <= t1)]
        if len(tr) < MIN_TRAIN_ROWS or len(te) < MIN_TEST_ROWS or tr["y"].nunique() < 2 or te["y"].nunique() < 2:
            continue
        p = predict_p(fit_classifier(tr[SETUP_FEATURES], tr["y"]), te[SETUP_FEATURES])
        y = te["y"].to_numpy(float)
        base = float(tr["y"].mean())
        ic = _ic_within_groups(p, te["R"].to_numpy(), te["setup"].to_numpy())
        if not np.isfinite(ic):
            continue
        ics.append(ic)
        aucs.append(_auc_within_groups(p, y, te["setup"].to_numpy()))
        skills.append(1.0 - np.mean((p - y) ** 2) / max(np.mean((base - y) ** 2), 1e-9))  # Brier skill vs base rate
        oos.append(pd.DataFrame({"setup": te["setup"].to_numpy(), "p": p, "y": y, "R": te["R"].to_numpy()}))

    k = len(ics)
    ic_mean = float(np.mean(ics)) if k else float("nan")
    auc_mean = float(np.nanmean(aucs)) if k else float("nan")
    t_stat = float("nan")
    if k >= 2 and np.std(ics, ddof=1) > 0:
        t_stat = ic_mean / (np.std(ics, ddof=1) / np.sqrt(k))
    trusted = bool(k >= MIN_FOLDS and ic_mean >= MIN_IC_EDGE and t_stat >= MIN_T_STAT)
    trust_weight = float(np.clip(t_stat / MIN_T_STAT, 0.0, 1.0)) if (k >= MIN_FOLDS and ic_mean > 0 and np.isfinite(t_stat)) else 0.0

    pooled = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=["setup", "p", "y", "R"])
    base_pooled = float(D["y"].mean())
    if len(pooled) >= 40 and pooled["y"].nunique() == 2:
        a, b = fit_platt(pooled["p"], pooled["y"])
    else:
        a, b = float(_logit(base_pooled)), 0.0

    per_setup = {}
    for s in setups.SETUPS:
        allr = D[D["setup"] == s]
        o = pooled[pooled["setup"] == s]
        per_setup[s] = {
            "n": int(len(allr)),
            "win_rate": float(allr["y"].mean()) if len(allr) else float("nan"),
            "mean_R": float(allr["R"].mean()) if len(allr) else float("nan"),
            "oos_n": int(len(o)),
            "oos_auc": _auc(o["p"], o["y"]) if len(o) >= 10 and o["y"].nunique() == 2 else float("nan"),
        }
    base_by_setup = {}
    for s, v in per_setup.items():
        if v["n"] > 0:
            base_by_setup[s] = float((v["n"] * v["win_rate"] + BASE_SHRINK_N * base_pooled) / (v["n"] + BASE_SHRINK_N))

    nw = D[D["y"] == 0]
    r_nw_pooled = float(nw["R"].mean()) if len(nw) else -1.0
    r_nonwin_by_setup = {}
    for s_ in setups.SETUPS:
        sub = nw[nw["setup"] == s_]
        if len(sub):
            r_nonwin_by_setup[s_] = float((len(sub) * sub["R"].mean() + BASE_SHRINK_N * r_nw_pooled) / (len(sub) + BASE_SHRINK_N))

    return {
        "r_nonwin_pooled": r_nw_pooled, "r_nonwin_by_setup": r_nonwin_by_setup,
        "n_rows": int(len(D)), "n_folds": k, "ic_by_fold": [round(x, 3) for x in ics],
        "ic_mean": ic_mean, "auc_mean": auc_mean, "t_stat": t_stat,
        "brier_skill_mean": float(np.mean(skills)) if skills else float("nan"),
        "trusted": trusted, "trust_weight": trust_weight,
        "calib_a": a, "calib_b": b,
        "base_pooled": base_pooled, "base_by_setup": base_by_setup, "per_setup": per_setup,
    }


def print_validation(m: dict, rules: dict, baselines: dict) -> None:
    rr = setups.TP_ATR / setups.SL_ATR
    logging.info(f"\n📊 Evidence 1 -- do the setups beat entering on a random day? (rule level, month-clustered, net of {setups.COST_PCT}% costs)")
    logging.info(f"   barriers: target {setups.TP_ATR:g} ATR / stop {setups.SL_ATR:g} ATR / {setups.MAX_HOLD} bars "
          f"(break-even win rate {1/(1+rr):.0%} before costs); any-day win rate: long {baselines['long']:.1%}, short {baselines['short']:.1%}")
    logging.info(f"   gate: n >= {MIN_RULE_N}, months >= {MIN_RULE_MONTHS}, lift >= {MIN_LIFT_R}R, t >= {RULE_MIN_T}, net R > 0")
    logging.info(f"   {'setup':<15} {'n':>4} {'win%':>6} {'anyday':>7} {'netR':>6} {'anyR':>6} {'liftR':>6} {'t':>5}  trusted")
    for s, v in rules.items():
        f = lambda x, fmt: (fmt % x) if np.isfinite(x) else "  n/a"
        logging.info(f"   {s:<15} {v['n']:>4} {f(v['win_rate']*100,'%5.1f'):>6} {f(v['any_win_rate']*100,'%6.1f'):>7} "
              f"{f(v['net_R'],'%+5.2f'):>6} {f(v['any_net_R'],'%+5.2f'):>6} {f(v['lift_R'],'%+5.2f'):>6} {f(v['t_lift'],'%4.1f'):>5}  "
              f"{'YES' if v['trusted'] else 'no'}")
    logging.info(f"\n📊 Evidence 2 -- can the model pick better instances WITHIN a setup? (walk-forward by date, purged)")
    logging.info(f"   labelled setups: {m['n_rows']}   scorable folds: {m['n_folds']} (need {MIN_FOLDS})   gate: mean IC >= {MIN_IC_EDGE}, t >= {MIN_T_STAT}")
    logging.info(f"   IC (prediction vs realised R, within setup) by fold: {m['ic_by_fold']}")
    logging.info(f"   mean IC {m['ic_mean']:+.3f}  t {m['t_stat']:.2f}  (win-rate AUC {m['auc_mean']:.3f}, Brier skill {m['brier_skill_mean']:+.3f}, "
          f"calibration slope {m['calib_b']:.2f})  trust {m['trust_weight']:.2f}  -> {'TRUSTED' if m['trusted'] else 'NOT trusted'}")


def rule_validation(D: pd.DataFrame, A: pd.DataFrame) -> dict:
    """Does each setup beat entering on a random day? (no model involved)

    The setup rules have no fitted parameters, so their whole history is a fair test. Results are in
    R (units of the stop distance), net of the cost assumption.

    Baseline: the unconditional mean net R of entering on ANY day in the same direction. (Matching by
    calendar month instead is biased: a breakout is caused by a big up-day, so months with breakouts
    are strong months, and random-day entries in them include days BEFORE the jump that ride it,
    while setup entries come only after it.)

    Significance: trades overlap in time and tickers move together, so the standard error is
    cluster-robust by calendar month (clusters are summed before squaring), not the naive n.
    """
    out = {}
    for s in setups.SETUPS:
        d = 1.0 if s in ("pullback_buy", "breakout_buy") else -1.0
        r, a = D[D["setup"] == s], A[A["direction"] == d]
        res = {"n": int(len(r)), "months": 0, "win_rate": float("nan"),
               "any_win_rate": float(a["y"].mean()) if len(a) else float("nan"),
               "net_R": float("nan"), "any_net_R": float("nan"), "lift_R": float("nan"), "t_lift": float("nan"),
               "trusted": False, "trust_weight": 0.0}
        if len(r) and len(a):
            net = lambda f: f["R"] - setups.COST_PCT / (setups.SL_ATR * f["atr_pct"])
            base = float(net(a).mean())
            e = (net(r) - base).to_numpy()
            ym = r["date"].dt.to_period("M").to_numpy()
            lift = float(e.mean())
            sums = pd.Series(e - lift).groupby(ym).sum()          # centred cluster sums
            m = len(sums)
            se = float(np.sqrt((sums ** 2).sum() * m / max(m - 1, 1)) / len(e)) if m >= 2 else float("nan")
            res.update(win_rate=float(r["y"].mean()), net_R=float(net(r).mean()), any_net_R=base,
                       months=int(m), lift_R=lift, t_lift=float(lift / se) if se and se > 0 else float("nan"))
            ok = np.isfinite(res["t_lift"])
            res["trusted"] = bool(ok and res["n"] >= MIN_RULE_N and m >= MIN_RULE_MONTHS
                                  and lift >= MIN_LIFT_R and res["t_lift"] >= RULE_MIN_T and res["net_R"] > 0)
            if ok and m >= MIN_RULE_MONTHS and lift > 0:
                res["trust_weight"] = float(np.clip(res["t_lift"] / RULE_MIN_T, 0.0, 1.0))
        out[s] = res
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_final(bundle: dict, x: dict, setup: str) -> dict:
    """Probability that THIS setup reaches its target before its stop, built from evidence in layers.

    p_any     win rate of entering on a random day in this direction (the no-edge anchor)
    p_setup   the setup's own historical win rate (shrunk toward the pooled rate when rare)
    p_anchor  p_any + u * (p_setup - p_any)         u = rule-level trust weight (0..1)
    p_model   the classifier, recalibrated out-of-sample (Platt); its deviation from p_setup is the
              within-setup adjustment, counted in proportion to the model trust weight w
    p_final = p_anchor + w * (p_cal - p_setup)

    With no evidence for either layer this collapses to the any-day win rate (no edge claimed).
    """
    m, rules = bundle["metrics"], bundle["rules"]
    is_long = setup in ("pullback_buy", "breakout_buy")
    p_any = bundle["baselines"]["long" if is_long else "short"]
    p_setup = float(m["base_by_setup"].get(setup, m["base_pooled"]))
    u = float(rules.get(setup, {}).get("trust_weight", 0.0))
    p_anchor = p_any + u * (p_setup - p_any)

    p_raw = p_cal = float("nan"); w = 0.0; adj = 0.0
    if bundle.get("model") is not None:
        X = pd.DataFrame([{c: float(x[c]) for c in SETUP_FEATURES}], columns=SETUP_FEATURES)
        p_raw = float(predict_p(bundle["model"], X)[0])
        p_cal = float(_sigmoid(m["calib_a"] + m["calib_b"] * _logit(p_raw)))
        w = float(m["trust_weight"])
        adj = w * (p_cal - p_setup)
    return {
        "p_any": p_any, "p_setup": p_setup, "p_raw": p_raw, "p_cal": p_cal,
        "setup_trust": u, "model_trust": w,
        "p_final": float(np.clip(p_anchor + adj, 0.01, 0.99)),
        "r_nonwin": float(m["r_nonwin_by_setup"].get(setup, m["r_nonwin_pooled"])),
        "validated": bool(rules.get(setup, {}).get("trusted", False) or m["trusted"]),
    }