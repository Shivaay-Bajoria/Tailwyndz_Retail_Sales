"""Baselines, the global LightGBM model and the cold-start fallback."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("models")


# ---------------------------------------------------------------- baselines
def baseline_seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Same week last year; falls back to the 13-week average when the series has no year-ago value."""
    return df["lag52"].fillna(df["o_mean13"]).to_numpy(float)


def baseline_moving_average(df: pd.DataFrame) -> np.ndarray:
    """Average of the last 13 weeks up to the forecast origin."""
    return df["o_mean13"].to_numpy(float)


BASELINES = {"seasonal_naive": baseline_seasonal_naive, "moving_avg_13": baseline_moving_average}


# ---------------------------------------------------------------- LightGBM
def make_lgbm(cfg: dict, objective: str | None = None, alpha: float | None = None,
              n_estimators: int | None = None) -> lgb.LGBMRegressor:
    p = dict(cfg["model"]["lgbm"])
    if objective:
        p["objective"] = objective
    if p["objective"] != "tweedie":
        p.pop("tweedie_variance_power", None)
    if p["objective"] == "quantile":
        p["alpha"] = alpha
    if n_estimators:
        p["n_estimators"] = n_estimators
    p.update(random_state=cfg["seed"], verbose=-1)
    return lgb.LGBMRegressor(**p)


def fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], cfg: dict, **kw):
    """Fit on `train` and return (model, non-negative predictions for `test`)."""
    m = make_lgbm(cfg, **kw)
    m.fit(train[feats], train["y"].astype(float))
    return m, np.clip(m.predict(test[feats]), 0, None)


def training_rows(sup: pd.DataFrame, cutoff: int, cfg: dict) -> pd.DataFrame:
    """Rows usable for training at a given cutoff: the target week must be <= cutoff (no leakage)."""
    m = (sup["target_week"] <= cutoff) & (~sup["flag_exclude"])
    if not cfg["data"]["train_on_inserted"]:
        m &= sup["flag_inserted"] == 0
    return sup[m]


# ---------------------------------------------------------------- cold start
def cold_start_forecast(pf: pd.DataFrame, targets: pd.DataFrame, cutoff: int, window: int = 13,
                        lo_q: float = 0.1, hi_q: float = 0.9) -> pd.DataFrame:
    """Forecast series that have no history at the origin (launched after the cutoff).

    Uses the units-per-weekly-basket rate of the same UPC across other stores over the last `window` weeks
    before the cutoff, scaled by the new store's average weekly baskets. Falls back to the category rate.
    The 10th/90th percentile of that rate gives a simple interval.
    """
    ref = pf[(pf.week_idx <= cutoff) & (pf.week_idx > cutoff - window) & (pf.flag_inserted == 0)].copy()
    ref["rate"] = ref["units"] / ref["avg_weekly_baskets"]
    ref["upc"] = ref["upc"].astype(str)
    ref["category"] = ref["category"].astype(str)
    stats = {}
    for key in ("upc", "category"):
        g = ref.groupby(key)["rate"]
        stats[key] = pd.DataFrame({"mean": g.mean(), "lo": g.quantile(lo_q), "hi": g.quantile(hi_q)})
    out = targets.copy()
    up, cat = out["upc"].astype(str), out["category"].astype(str)
    for name in ("mean", "lo", "hi"):
        rate = up.map(stats["upc"][name]).fillna(cat.map(stats["category"][name]))
        out[f"cs_{name}"] = (rate * out["avg_weekly_baskets"]).astype(float)
    out["forecast"] = out["cs_mean"].fillna(0).clip(lower=0)
    out["lower_raw"] = out["cs_lo"].fillna(0).clip(lower=0)
    out["upper_raw"] = out["cs_hi"].fillna(out["forecast"]).clip(lower=out["forecast"])
    return out.drop(columns=["cs_mean", "cs_lo", "cs_hi"])
