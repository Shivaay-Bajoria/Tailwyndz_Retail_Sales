"""Feature-group ablation: what does each block of information add?"""
from __future__ import annotations

import pandas as pd

from .features import FEATURE_GROUPS, feature_list
from .models import fit_predict_lgbm, training_rows
from .utils import get_logger, metric_row
from .validation import eval_rows

log = get_logger("ablation")


def ablation_sets() -> list[tuple[str, str, list[str]]]:
    """(label, kind, groups). History is always included; it carries the origin-level signal."""
    full = list(FEATURE_GROUPS)
    inc, groups = [], ["history"]
    out = [("history only", "incremental", ["history"])]
    for g in ["price", "promo", "product", "store", "calendar"]:
        groups = groups + [g]
        out.append((f"+ {g}", "incremental", list(groups)))
    for g in ["price", "promo", "product", "store", "calendar"]:
        out.append((f"full minus {g}", "leave_one_out", [x for x in full if x != g]))
    return out


def run_ablation(sup: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    acfg = cfg["ablation"]
    rows = []
    for label, kind, groups in ablation_sets():
        feats = feature_list(groups)
        for origin in acfg["origins"]:
            train = training_rows(sup, origin, cfg)
            if acfg.get("max_train_rows") and len(train) > acfg["max_train_rows"]:
                train = train.sample(acfg["max_train_rows"], random_state=cfg["seed"])
            test = eval_rows(sup, origin)
            _, pred = fit_predict_lgbm(train, test, feats, cfg, n_estimators=acfg["n_estimators"])
            rows.append({"feature_set": label, "kind": kind, "groups": "+".join(groups), "origin": origin,
                         **metric_row(test["y"], pred)})
        log.info("ablation: %s done", label)
    res = pd.DataFrame(rows)
    mean = res.groupby(["feature_set", "kind", "groups"], as_index=False, sort=False)[["wmape", "rmse", "bias"]].mean()
    full = mean.loc[mean.feature_set == "+ calendar", "wmape"].iloc[0]
    mean["origin"] = "mean"
    mean["delta_wmape_vs_full"] = mean["wmape"] - full
    return pd.concat([res, mean], ignore_index=True)
