"""Prediction intervals: LightGBM quantile regression with split-conformal (CQR) calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ALL_FEATURES
from .models import make_lgbm, training_rows
from .utils import get_logger
from .validation import eval_rows

log = get_logger("intervals")


def fit_quantile_models(train: pd.DataFrame, feats: list[str], cfg: dict):
    a, n = cfg["intervals"]["alpha"], cfg["intervals"]["quantile_estimators"]
    lo = make_lgbm(cfg, "quantile", a / 2, n).fit(train[feats], train["y"].astype(float))
    hi = make_lgbm(cfg, "quantile", 1 - a / 2, n).fit(train[feats], train["y"].astype(float))
    return lo, hi


def predict_bounds(models, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.clip(models[0].predict(X), 0, None), np.clip(models[1].predict(X), 0, None)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def conformal_qhat(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Conformalised quantile regression: the margin that restores (1 - alpha) coverage on calibration data."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def add_intervals(pf: pd.DataFrame, sup: pd.DataFrame, origins: dict, holdout: pd.DataFrame, cfg: dict):
    """Attach interval bounds to the holdout forecasts.

    Calibration uses the most recent CV fold: quantile models trained only on data up to that fold's origin
    produce the conformity scores that widen (or tighten) the holdout bounds.
    Returns (holdout_with_intervals, coverage_table, qhat).
    """
    alpha = cfg["intervals"]["alpha"]
    feats = ALL_FEATURES
    calib_o, hold_o = origins["cv"][0], origins["holdout"]

    calib_models = fit_quantile_models(training_rows(sup, calib_o, cfg), feats, cfg)
    calib = eval_rows(sup, calib_o)
    clo, chi = predict_bounds(calib_models, calib[feats])
    qhat = conformal_qhat(calib["y"].to_numpy(float), clo, chi, alpha)
    log.info("Conformal margin from calibration fold (origin %d): %.3f units", calib_o, qhat)

    hold_models = fit_quantile_models(training_rows(sup, hold_o, cfg), feats, cfg)
    test = eval_rows(sup, hold_o)
    lo, hi = predict_bounds(hold_models, test[feats])
    out = holdout.copy()
    assert len(out) == len(test), "holdout forecasts and interval rows must align"
    out["lower_raw"], out["upper_raw"] = lo, hi
    out["lower"] = np.clip(lo - qhat, 0, None)
    out["upper"] = hi + qhat
    # keep the point forecast inside its own interval
    out["lower"] = np.minimum(out["lower"], out["lgbm"])
    out["upper"] = np.maximum(out["upper"], out["lgbm"])
    out["lower_raw"] = np.minimum(out["lower_raw"], out["lgbm"])
    out["upper_raw"] = np.maximum(out["upper_raw"], out["lgbm"])
    return out, coverage_table(out, 1 - alpha), qhat


def coverage_table(df: pd.DataFrame, nominal: float) -> pd.DataFrame:
    """Empirical coverage and mean width, overall and by horizon / category / promo type."""
    rows = []

    def add(name, seg, g):
        for kind, lo, hi in (("raw_quantile", "lower_raw", "upper_raw"), ("conformal", "lower", "upper")):
            rows.append({"by": name, "segment": str(seg), "interval": kind, "nominal": nominal, "n": len(g),
                         "coverage": float(((g["actual"] >= g[lo]) & (g["actual"] <= g[hi])).mean()),
                         "mean_width": float((g[hi] - g[lo]).mean())})

    add("overall", "all", df)
    for by in ("h", "category", "promo_type", "tier"):
        for seg, g in df.groupby(by, observed=True):
            add(by, seg, g)
    return pd.DataFrame(rows)
