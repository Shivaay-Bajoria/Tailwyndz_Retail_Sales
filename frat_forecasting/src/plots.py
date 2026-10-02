"""Figures written to outputs/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def weekly_trend(pf: pd.DataFrame, path: Path):
    d = pf[pf.flag_inserted == 0].assign(category=lambda x: x.category.astype(str))
    t = d.groupby(["week_end_date", "category"])["units"].sum().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    t.plot(ax=ax, lw=1.2)
    ax.set_title("Weekly units by category (all stores)")
    ax.set_xlabel("")
    ax.set_ylabel("units")
    _save(fig, path)


def cv_metrics(summary: pd.DataFrame, path: Path):
    d = summary[(summary.outliers == "included") & summary.fold.isin(["cv_mean", "holdout"])]
    piv = d.pivot(index="model", columns="fold", values="wmape").loc[["lgbm", "moving_avg_13", "seasonal_naive"]]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    piv[["cv_mean", "holdout"]].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("WMAPE (lower is better)")
    ax.set_title("Forecast accuracy: LightGBM vs baselines")
    _save(fig, path)


def forecast_vs_actual(h: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    w = h.groupby("week_end_date")[["actual", "forecast"]].sum()
    axes[0].plot(w.index, w["actual"], marker="o", label="actual")
    axes[0].plot(w.index, w["forecast"], marker="o", label="forecast")
    axes[0].set_title("Holdout: total weekly units")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)
    # six high-volume example series, one panel overlaying the interval of the largest
    vol = h.groupby("series_id")["actual"].sum().sort_values(ascending=False)
    cats = h.drop_duplicates("series_id").set_index("series_id")["category"].astype(str)
    picks, seen = [], set()
    for sid in vol.index:
        if cats[sid] not in seen:
            picks.append(sid)
            seen.add(cats[sid])
        if len(picks) == 4:
            break
    ax = axes[1]
    for sid in picks:
        g = h[h.series_id == sid].sort_values("week_end_date")
        line = ax.plot(g["week_end_date"], g["actual"], marker="o", ms=3, lw=1, label=f"{cats[sid].title()} (actual)")[0]
        ax.plot(g["week_end_date"], g["forecast"], ls="--", lw=1, color=line.get_color())
        ax.fill_between(g["week_end_date"], g["lower"], g["upper"], alpha=0.15, color=line.get_color())
    ax.set_title("Largest series per category: actual, forecast (dashed), 80% interval")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7)
    _save(fig, path)


def group_importance(gi: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    d = gi.sort_values("share")
    ax.barh(d["group"], d["share"])
    ax.set_xlabel("share of mean |SHAP|")
    ax.set_title("What drives the forecasts")
    _save(fig, path)


def ablation(abl: pd.DataFrame, path: Path):
    m = abl[abl.origin == "mean"]
    inc = m[m.kind == "incremental"]
    loo = m[m.kind == "leave_one_out"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(inc["feature_set"], inc["wmape"], marker="o")
    axes[0].set_title("Adding feature groups")
    axes[0].set_ylabel("WMAPE")
    axes[0].tick_params(axis="x", rotation=40)
    axes[1].barh(loo["feature_set"], loo["delta_wmape_vs_full"])
    axes[1].set_title("WMAPE change when a group is removed")
    _save(fig, path)


def elasticity(el: pd.DataFrame, path: Path):
    d = el[(el.level == "category")]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(d.group.nunique())
    for i, spec in enumerate(["A", "B"]):
        s = d[d.spec == spec].set_index("group")
        ax.errorbar(s["elasticity"], y + (i - 0.5) * 0.25, xerr=[s["elasticity"] - s["ci_low"], s["ci_high"] - s["elasticity"]],
                    fmt="o", capsize=3, label=f"spec {spec}")
    ax.set_yticks(y, d[d.spec == "A"]["group"])
    ax.axvline(-1, color="grey", lw=0.8, ls=":")
    ax.set_title("Price elasticity by category (95% CI)")
    ax.legend()
    _save(fig, path)


def promo_lift(pl: pd.DataFrame, path: Path):
    d = pl[(pl.split == "category") & (pl.promo != "none")]
    piv = d.pivot(index="segment", columns="promo", values="units_lift")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    piv.plot.bar(ax=ax, rot=15)
    ax.set_ylabel("median units vs own non-promo weeks (x)")
    ax.set_title("Promotion lift by category")
    _save(fig, path)


def scenarios(summary: pd.DataFrame, path: Path):
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    yy = np.arange(len(s))
    ax.barh(yy - 0.2, s["units_change_pct"] * 100, height=0.4, label="units")
    ax.barh(yy + 0.2, s["revenue_change_pct"] * 100, height=0.4, label="revenue")
    ax.set_yticks(yy, s["scenario"])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("% change vs baseline forecast, 13 holdout weeks")
    ax.legend()
    _save(fig, path)


def coverage(cov: pd.DataFrame, nominal: float, path: Path):
    d = cov[(cov.by == "h")].assign(h=lambda x: x.segment.astype(int)).sort_values("h")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for kind, g in d.groupby("interval"):
        ax.plot(g["h"], g["coverage"], marker="o", label=kind)
    ax.axhline(nominal, color="grey", ls=":")
    ax.set_xlabel("forecast horizon (weeks)")
    ax.set_ylabel("empirical coverage")
    ax.set_title("80% interval coverage on the holdout")
    ax.legend()
    _save(fig, path)
