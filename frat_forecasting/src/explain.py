"""Driver attribution from the fitted LightGBM model using TreeSHAP (LightGBM's pred_contrib)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_GROUPS
from .utils import get_logger

log = get_logger("explain")

FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature contributions to the prediction (raw model scale, log link for Tweedie)."""
    contrib = model.booster_.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def driver_attribution(model, holdout_rows: pd.DataFrame, feats: list[str], cfg: dict):
    """Return (feature_importance, group_importance, group_by_category, local_explanations)."""
    n = min(cfg["explain"]["sample_rows"], len(holdout_rows))
    sample = holdout_rows.sample(n, random_state=cfg["seed"])
    sv = shap_values(model, sample[feats])
    fi = sv.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    fi.columns = ["feature", "mean_abs_shap"]
    fi["group"] = fi["feature"].map(FEATURE_TO_GROUP)
    fi["share"] = fi["mean_abs_shap"] / fi["mean_abs_shap"].sum()

    gi = fi.groupby("group", as_index=False)[["mean_abs_shap", "share"]].sum().sort_values("share", ascending=False)

    grp_sv = sv.T.groupby(FEATURE_TO_GROUP).sum().T.abs()
    by_cat = grp_sv.groupby(sample["category"].astype(str).to_numpy()).mean()
    by_cat = by_cat.div(by_cat.sum(axis=1), axis=0).reset_index(names="category")

    # local explanations for the five largest forecasts in the sample
    top = sample.assign(_pred=model.predict(sample[feats])).nlargest(5, "_pred")
    rows = []
    for idx, r in top.iterrows():
        contrib = sv.loc[idx].reindex(sv.loc[idx].abs().sort_values(ascending=False).index).head(5)
        rows.append({"store_num": r["store_num"], "upc": str(r["upc"]), "week_end_date": r["week_end_date"],
                     "forecast": float(r["_pred"]),
                     "top_drivers": "; ".join(f"{k} ({v:+.2f})" for k, v in contrib.items())})
    return fi, gi, by_cat, pd.DataFrame(rows)
