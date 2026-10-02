"""Price elasticity and promotion lift.

Elasticity = coefficient on log(price) in a log-log regression of units on price with series (store x UPC)
and week fixed effects, so it is identified from within-series price changes after removing chain-wide
weekly shocks (seasonality, holidays). Two specifications are reported:

  A  price only          -> total price response, including the effect of temporary price reductions
  B  price + promo flags -> price response holding feature / display / sale-tag status fixed

These are observational estimates, not causal ones (promotions are not randomly assigned).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import get_logger

log = get_logger("elasticity")

PROMO_COLS = ["feature", "display", "tpr_only"]


def _sample(pf: pd.DataFrame) -> pd.DataFrame:
    d = pf[(pf.flag_inserted == 0) & (~pf.flag_exclude) & (pf.units > 0) & (pf.price > 0)].copy()
    d["log_units"] = np.log(d["units"])
    d["log_price"] = np.log(d["price"])
    return d


def _demean(df: pd.DataFrame, cols: list[str], iters: int = 8) -> pd.DataFrame:
    """Absorb series and week fixed effects by alternating projections."""
    x = df[cols].astype(float).copy()
    for _ in range(iters):
        for fe in ("series_id", "week_idx"):
            x = x - x.groupby(df[fe].to_numpy()).transform("mean")
    return x


def _fit(df: pd.DataFrame, spec: str) -> dict | None:
    xcols = ["log_price"] + (PROMO_COLS if spec == "B" else [])
    xcols = [c for c in xcols if df[c].nunique() > 1]
    if len(df) < 200 or "log_price" not in xcols:
        return None
    z = _demean(df, ["log_units", *xcols])
    res = sm.OLS(z["log_units"], z[xcols]).fit(cov_type="cluster", cov_kwds={"groups": df["series_id"].to_numpy()})
    b, se = res.params["log_price"], res.bse["log_price"]
    return {"elasticity": b, "std_err": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se, "n": int(len(df)),
            "price_cv_within": float(df.groupby("series_id")["log_price"].std().mean())}


def estimate_elasticities(pf: pd.DataFrame) -> pd.DataFrame:
    d = _sample(pf)
    d["category"] = d["category"].astype(str)
    d["tier"] = d["tier"].astype(str)
    rows = []

    def add(level, name, sub):
        for spec in ("A", "B"):
            r = _fit(sub, spec)
            if r:
                rows.append({"level": level, "group": name, "spec": spec, **r})

    add("overall", "all", d)
    for cat, g in d.groupby("category"):
        add("category", cat, g)
        for tier, gt in g.groupby("tier"):
            add("category x tier", f"{cat} | {tier}", gt)
    desc = pf.drop_duplicates("upc").set_index(pf.drop_duplicates("upc")["upc"].astype(str))["category"].astype(str)
    for upc, g in d.groupby(d["upc"].astype(str)):
        add("upc", f"{upc} ({desc.get(upc, '')})", g)
    out = pd.DataFrame(rows)
    out["elastic"] = out["elasticity"] < -1
    log.info("Elasticity: %d regressions", len(out))
    return out


def promo_lift(pf: pd.DataFrame) -> pd.DataFrame:
    """Average lift versus each series' own non-promoted weeks, by category and promotion type."""
    d = pf[(pf.flag_inserted == 0) & (~pf.flag_exclude)].copy()
    d["category"] = d["category"].astype(str)
    base = d[d.promo_type == 0].groupby("series_id")[["units", "visits"]].mean().rename(columns=lambda c: f"base_{c}")
    d = d.join(base, on="series_id")
    d = d[(d.base_units > 0) & (d.base_visits > 0)]
    d["units_index"] = d["units"] / d["base_units"]
    d["visits_index"] = d["visits"] / d["base_visits"]
    d["units_per_visit"] = d["units"] / d["visits"].replace(0, np.nan)
    names = {0: "none", 1: "TPR only", 2: "display only", 3: "feature only", 4: "feature + display"}
    d["promo"] = d["promo_type"].map(names)
    d["tier"] = d["tier"].astype(str)
    out = []
    for by in ("category", "tier"):
        g = d.groupby([by, "promo"]).agg(rows=("units", "size"), units_lift=("units_index", "median"),
                                          visits_lift=("visits_index", "median"),
                                          units_per_visit=("units_per_visit", "median"),
                                          avg_discount=("discount_pct", "mean")).reset_index()
        g.insert(0, "split", by)
        out.append(g.rename(columns={by: "segment"}))
    allg = d.groupby("promo").agg(rows=("units", "size"), units_lift=("units_index", "median"),
                                  visits_lift=("visits_index", "median"),
                                  units_per_visit=("units_per_visit", "median"),
                                  avg_discount=("discount_pct", "mean")).reset_index()
    allg.insert(0, "segment", "all")
    allg.insert(0, "split", "overall")
    return pd.concat([allg, *out], ignore_index=True)
