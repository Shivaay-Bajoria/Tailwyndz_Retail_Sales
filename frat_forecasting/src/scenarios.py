"""Price and promotion what-if simulations for the 13 holdout weeks.

Two methods are used, because they answer different questions and the data supports them differently:

* Everyday price changes (+/- x%) use the category price elasticity from the fixed-effects log-log regression
  (specification B: promotion status held fixed). The tree model is NOT used for these: in this data a price
  cut never occurs without a promotion flag, so the model has no basis for such a counterfactual and responds
  unreliably (it can even predict falling sales after a cut).
* Promotion scenarios (sale tag of a given depth, display, circular feature) change the flags and price in line
  with combinations that do occur in the data, so the fitted LightGBM model is used.

The sale-tag results are cross-checked against the elasticity (specification A, total price response).
All results are observational model-based simulations, not causal estimates. Revenue = shelf price x units;
no cost data exists, so profit is out of scope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ALL_FEATURES
from .utils import get_logger

log = get_logger("scenarios")


def _set_promo(df: pd.DataFrame) -> pd.DataFrame:
    df["promo_type"] = np.select([(df.feature == 1) & (df.display == 1), df.feature == 1, df.display == 1, df.tpr_only == 1],
                                 [4, 3, 2, 1], default=0)
    return df


def apply_sale_tag(df: pd.DataFrame, depth: float) -> tuple[pd.DataFrame, pd.Series]:
    """Sale tag of `depth` on every row without any promotion. Returns (modified rows, changed mask)."""
    changed = df["promo_type"] == 0
    new_price = df["price"].where(~changed, df["base_price"] * (1 - depth))
    df["price_gap_cat"] = df["price_gap_cat"] * (new_price / df["price"])
    df["price"] = new_price
    df.loc[changed, "tpr_only"] = 1
    df["discount_pct"] = (1 - df["price"] / df["base_price"]).clip(lower=0, upper=1)
    return _set_promo(df), changed


def apply_display(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    changed = df["promo_type"].isin([0, 1])
    df.loc[changed, "display"] = 1
    return _set_promo(df), changed


def apply_feature(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    changed = df["promo_type"].isin([0, 1])
    df.loc[changed, "feature"] = 1
    return _set_promo(df), changed


def apply_feature_display(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    changed = df["promo_type"].isin([0, 1, 2, 3])
    df.loc[changed, ["feature", "display"]] = 1
    return _set_promo(df), changed


def _predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(df[ALL_FEATURES]), 0, None)


def run_scenarios(model, rows: pd.DataFrame, elasticities: pd.DataFrame, cfg: dict):
    """Returns (summary, by_category, by_tier, cross_check)."""
    base = rows.copy()
    base["pred"] = _predict(model, base)
    base["cat_label"] = base["category"].astype(str)   # label columns keep the model's categorical features untouched
    base["tier_label"] = base["tier"].astype(str)
    el = elasticities[elasticities.level == "category"]
    el_b = el[el.spec == "B"].set_index("group")["elasticity"]
    el_a = el[el.spec == "A"].set_index("group")["elasticity"]

    summary, by_cat, by_tier, check = [], [], [], []
    base_rev = base["pred"] * base["price"]
    totals = {"units": base["pred"].sum(), "revenue": base_rev.sum()}
    base_by = {k: pd.DataFrame({"units": base["pred"], "revenue": base_rev, "k": base[k]}).groupby("k")[["units", "revenue"]].sum()
               for k in ("cat_label", "tier_label")}

    def record(name: str, method: str, pred: np.ndarray, price: pd.Series, changed: pd.Series):
        rev = pred * price
        tot_u, tot_r = pred.sum(), rev.sum()
        ch = changed.to_numpy()
        summary.append({"scenario": name, "method": method, "rows_changed": int(ch.sum()), "rows_total": len(base),
                        "base_units": totals["units"], "scenario_units": tot_u,
                        "units_change_pct": tot_u / totals["units"] - 1,
                        "units_change_pct_on_changed_rows": float(pred[ch].sum() / base["pred"].to_numpy()[ch].sum() - 1) if ch.any() else np.nan,
                        "base_revenue": totals["revenue"], "scenario_revenue": tot_r,
                        "revenue_change_pct": tot_r / totals["revenue"] - 1})
        for key, out_key, store in (("cat_label", "category", by_cat), ("tier_label", "tier", by_tier)):
            s = pd.DataFrame({"units": pred, "revenue": rev.to_numpy(), "k": base[key].to_numpy()}).groupby("k")[["units", "revenue"]].sum()
            b = base_by[key]
            store.append(pd.DataFrame({"scenario": name, "method": method, out_key: s.index,
                                       "units_change_pct": (s["units"] / b["units"] - 1).to_numpy(),
                                       "revenue_change_pct": (s["revenue"] / b["revenue"] - 1).to_numpy()}))

    # 1) everyday price changes: elasticity-based
    for pct in cfg["scenarios"]["price_change_pcts"]:
        e = base["cat_label"].map(el_b).fillna(el_b.mean()).to_numpy()
        pred = base["pred"].to_numpy() * (1 + pct) ** e
        record(f"everyday price {pct:+.0%}", "elasticity (spec B, by category)", pred, base["price"] * (1 + pct),
               pd.Series(True, index=base.index))

    # 2) promotions: model-based
    def model_scenario(name, fn, **kw):
        sc, changed = fn(base.copy(), **kw)
        pred = _predict(model, sc)
        record(name, "LightGBM", pred, sc["price"], changed)
        return sc, changed, pred

    for d in cfg["scenarios"]["sale_tag_depths"]:
        sc, changed, pred = model_scenario(f"sale tag {d:.0%} off (un-promoted rows)", apply_sale_tag, depth=d)
        # cross-check on the changed rows: model vs elasticity A (total price response)
        ratio = (sc["price"] / base["price"]).to_numpy()
        for cat in sorted(base["cat_label"].unique()):
            m = (changed & (base["cat_label"] == cat)).to_numpy()
            if not m.any() or cat not in el_a.index:
                continue
            check.append({"scenario": f"sale tag {d:.0%} off", "category": cat, "rows": int(m.sum()),
                          "model_units_change_pct": pred[m].sum() / base["pred"].to_numpy()[m].sum() - 1,
                          "elasticity_implied_pct": float((base["pred"].to_numpy()[m] * ratio[m] ** el_a[cat]).sum()
                                                          / base["pred"].to_numpy()[m].sum() - 1),
                          "category_elasticity_A": float(el_a[cat])})
    model_scenario("add display (no price cut)", apply_display)
    model_scenario("add circular feature (no price cut)", apply_feature)
    model_scenario("feature + display (no price cut)", apply_feature_display)

    log.info("Scenarios: %d simulated", len(summary))
    return (pd.DataFrame(summary), pd.concat(by_cat, ignore_index=True), pd.concat(by_tier, ignore_index=True),
            pd.DataFrame(check))
