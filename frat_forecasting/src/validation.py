"""Rolling-origin validation: baselines versus the global LightGBM model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ALL_FEATURES
from .models import BASELINES, cold_start_forecast, fit_predict_lgbm, training_rows
from .utils import get_logger, metric_row, segment_metrics

log = get_logger("validation")

META = ["series_id", "store_num", "upc", "origin", "target_week", "week_end_date", "h", "category", "sub_category",
        "tier", "state", "promo_type", "history_bucket", "flag_outlier", "price", "base_price", "units_ref"]


def eval_rows(sup: pd.DataFrame, origin: int) -> pd.DataFrame:
    """Rows scored at an origin: observed (not inserted) and trustworthy weeks only."""
    m = (sup["origin"] == origin) & (sup["flag_inserted"] == 0) & (~sup["flag_exclude"])
    return sup[m]


def cold_start_targets(pf: pd.DataFrame, sup: pd.DataFrame, origin: int, horizon: int) -> pd.DataFrame:
    """Observed rows in the forecast window whose series has no row at the origin (no history yet)."""
    win = pf[(pf.week_idx > origin) & (pf.week_idx <= origin + horizon) & (pf.flag_inserted == 0) & (~pf.flag_exclude)]
    have = set(zip(sup.loc[sup.origin == origin, "series_id"], sup.loc[sup.origin == origin, "target_week"]))
    keep = [(s, w) not in have for s, w in zip(win.series_id, win.week_idx)]
    out = win[keep].copy()
    out["origin"] = origin
    out["h"] = out["week_idx"] - origin
    return out.rename(columns={"week_idx": "target_week"})


def run_fold(pf: pd.DataFrame, sup: pd.DataFrame, origin: int, horizon: int, cfg: dict,
             feats: list[str] | None = None, label: str = "") -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Train on targets <= origin and forecast the next `horizon` weeks. Returns (scored, cold_start, model)."""
    feats = feats or ALL_FEATURES
    train = training_rows(sup, origin, cfg)
    test = eval_rows(sup, origin)
    model, pred = fit_predict_lgbm(train, test, feats, cfg)
    out = test[[c for c in META if c in test.columns and c != "units_ref"]].copy()
    out["actual"] = test["y"].to_numpy(float)
    out["lgbm"] = pred
    for name, fn in BASELINES.items():
        out[name] = fn(test)
    out["fold"] = label or f"origin_{origin}"
    cold = cold_start_targets(pf, sup, origin, horizon)
    if len(cold):
        cold = cold_start_forecast(pf, cold, origin)
        cold["actual"] = cold["units"]
        cold["fold"] = out["fold"].iloc[0]
    log.info("%s: train=%d rows, scored=%d, cold-start=%d", out["fold"].iloc[0], len(train), len(out), len(cold))
    return out, cold, model


def run_validation(pf: pd.DataFrame, sup: pd.DataFrame, origins: dict, horizon: int, cfg: dict):
    """Run every CV fold plus the holdout. Returns (all_scored, cold_all, holdout_model, summary)."""
    folds = [(o, f"cv{i + 1}") for i, o in enumerate(origins["cv"])] + [(origins["holdout"], "holdout")]
    scored, colds, model = [], [], None
    for o, label in folds:
        s, c, m = run_fold(pf, sup, o, horizon, cfg, label=label)
        scored.append(s)
        colds.append(c)
        if label == "holdout":
            model = m
    all_scored = pd.concat(scored, ignore_index=True)
    cold_all = pd.concat([c for c in colds if len(c)], ignore_index=True) if any(len(c) for c in colds) else pd.DataFrame()
    return all_scored, cold_all, model, summarise(all_scored)


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """WMAPE / RMSE / bias per fold and model, plus the CV average."""
    models = ["lgbm", *BASELINES]
    rows = []
    for fold, g in scored.groupby("fold", sort=False):
        for m in models:
            rows.append({"fold": fold, "model": m, "outliers": "included", **metric_row(g["actual"], g[m])})
            gc = g[~g["flag_outlier"]]
            rows.append({"fold": fold, "model": m, "outliers": "excluded", **metric_row(gc["actual"], gc[m])})
    res = pd.DataFrame(rows)
    cv = res[res.fold.str.startswith("cv")].groupby(["model", "outliers"], as_index=False)[["wmape", "rmse", "bias"]].mean()
    cv["fold"], cv["n"] = "cv_mean", np.nan
    return pd.concat([res, cv[res.columns]], ignore_index=True)


def segment_report(holdout: pd.DataFrame, cold: pd.DataFrame | None = None) -> pd.DataFrame:
    """Holdout metrics by segment for the LightGBM model and the best simple baseline."""
    rows = []
    for by in ["category", "sub_category", "tier", "state", "promo_type", "history_bucket", "h"]:
        for model in ["lgbm", "moving_avg_13"]:
            seg = segment_metrics(holdout, by, "actual", model)
            seg.insert(0, "model", model)
            seg.insert(0, "segment_by", by)
            seg = seg.rename(columns={by: "segment"})
            seg["segment"] = seg["segment"].astype(str)
            rows.append(seg)
    if cold is not None and len(cold):
        rows.append(pd.DataFrame([{"segment_by": "history_bucket", "model": "cold_start_rule", "segment": "no history",
                                   **metric_row(cold["actual"], cold["forecast"])}]))
    return pd.concat(rows, ignore_index=True)
