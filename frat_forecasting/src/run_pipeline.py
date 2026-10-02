"""Run the full pipeline end to end.

    python -m src.run_pipeline --config config.yaml [--fast] [--skip-ablation] [--reload]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import plots
from .ablation import run_ablation
from .data_loading import load_tables, merge_tables, prepare_lookups
from .elasticity import estimate_elasticities, promo_lift
from .explain import driver_attribution
from .features import ALL_FEATURES, build_panel_features, build_supervised, make_origins
from .intervals import add_intervals
from .panel import build_panel
from .quality_checks import run as run_quality
from .scenarios import run_scenarios
from .utils import get_logger, load_config, metric_row, resolve
from .validation import eval_rows, run_validation, segment_report

log = get_logger("pipeline")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fast", action="store_true", help="fewer trees and ablation folds, for a quick smoke test")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--reload", action="store_true", help="re-read the xlsx instead of using the parquet cache")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.fast:
        cfg["model"]["lgbm"]["n_estimators"] = 60
        cfg["intervals"]["quantile_estimators"] = 40
        cfg["ablation"].update(n_estimators=40, origins=[142])
    out = resolve(cfg, "output_dir")
    fig_dir, drv_dir, scn_dir = out / "figures", out / "driver_attribution", out / "scenarios"
    for d in (fig_dir, drv_dir, scn_dir):
        d.mkdir(parents=True, exist_ok=True)
    horizon, t0 = cfg["horizon"], time.time()

    # 1. data -----------------------------------------------------------------
    tables = load_tables(cfg, force=args.reload)
    products, stores = prepare_lookups(tables["products"], tables["stores"])
    tables["products"], tables["stores"] = products, stores
    clean, _ = run_quality(cfg, tables)
    panel, week_map = build_panel(clean, cfg)
    pf = build_panel_features(panel, products, stores)
    pf.to_parquet(out / "modelling_dataset.parquet", index=False)

    # 2. supervised rows and validation ------------------------------------------
    origins = make_origins(pf["week_idx"].max(), horizon, cfg["validation"]["origin_stride"],
                           cfg["validation"]["min_origin"], cfg["validation"]["n_cv_folds"])
    log.info("Origins: holdout=%d, cv=%s", origins["holdout"], origins["cv"])
    sup = build_supervised(pf, origins["all"], horizon)
    scored, cold, model, summary = run_validation(pf, sup, origins, horizon, cfg)
    summary.to_csv(out / "cv_results.csv", index=False)

    holdout = scored[scored.fold == "holdout"].reset_index(drop=True)
    cold_h = cold[cold.fold == "holdout"].reset_index(drop=True) if len(cold) else pd.DataFrame()
    segment_report(holdout, cold_h).to_csv(out / "segment_metrics.csv", index=False)

    # 3. intervals and the forecast file --------------------------------------------
    holdout_i, cov, qhat = add_intervals(pf, sup, origins, holdout, cfg)
    cov.to_csv(out / "interval_coverage.csv", index=False)
    cols = ["week_end_date", "store_num", "upc", "category", "h", "forecast", "lower", "upper", "lower_raw", "upper_raw",
            "actual", "method", "history_bucket", "promo_type", "flag_outlier", "series_id"]
    f = holdout_i.rename(columns={"lgbm": "forecast"}).assign(method="lgbm_global")
    parts = [f]
    if len(cold_h):
        c = cold_h.rename(columns={"upc": "upc"}).assign(method="cold_start_rule", history_bucket="no history")
        c["lower"], c["upper"] = c["lower_raw"], c["upper_raw"]
        parts.append(c)
    forecasts = pd.concat(parts, ignore_index=True)
    forecasts["category"] = forecasts["category"].astype(str)
    forecasts[[c for c in cols if c in forecasts.columns]].sort_values(["store_num", "upc", "week_end_date"]).to_csv(
        out / "forecasts.csv", index=False)

    # 4. elasticity and promo lift --------------------------------------------------
    el = estimate_elasticities(pf)
    el.to_csv(out / "elasticity_estimates.csv", index=False)
    pl = promo_lift(pf)
    pl.to_csv(out / "promo_lift.csv", index=False)

    # 5. ablation -------------------------------------------------------------------
    abl = None
    if cfg["ablation"]["enabled"] and not args.skip_ablation:
        abl = run_ablation(sup, cfg)
        abl.to_csv(out / "ablation_results.csv", index=False)

    # 6. driver attribution ---------------------------------------------------------
    hold_rows = eval_rows(sup, origins["holdout"])
    fi, gi, by_cat, local = driver_attribution(model, hold_rows, ALL_FEATURES, cfg)
    fi.to_csv(drv_dir / "feature_importance_shap.csv", index=False)
    gi.to_csv(drv_dir / "group_importance.csv", index=False)
    by_cat.to_csv(drv_dir / "group_importance_by_category.csv", index=False)
    local.to_csv(drv_dir / "local_explanations.csv", index=False)

    # 7. scenarios ------------------------------------------------------------------
    s_sum, s_cat, s_tier, s_chk = run_scenarios(model, hold_rows, el, cfg)
    s_sum.to_csv(scn_dir / "scenario_summary.csv", index=False)
    s_cat.to_csv(scn_dir / "scenario_by_category.csv", index=False)
    s_tier.to_csv(scn_dir / "scenario_by_tier.csv", index=False)
    s_chk.to_csv(scn_dir / "scenario_elasticity_check.csv", index=False)

    # 8. figures --------------------------------------------------------------------
    plots.weekly_trend(pf, fig_dir / "weekly_trend.png")
    plots.cv_metrics(summary, fig_dir / "cv_metrics.png")
    plots.forecast_vs_actual(f, fig_dir / "forecast_vs_actual.png")
    plots.group_importance(gi, fig_dir / "group_importance.png")
    plots.elasticity(el, fig_dir / "elasticity_by_category.png")
    plots.promo_lift(pl, fig_dir / "promo_lift.png")
    plots.scenarios(s_sum, fig_dir / "scenarios.png")
    plots.coverage(cov, 1 - cfg["intervals"]["alpha"], fig_dir / "interval_coverage.png")
    if abl is not None:
        plots.ablation(abl, fig_dir / "ablation.png")

    # 9. run summary ----------------------------------------------------------------
    win = pf[(pf.week_idx > origins["holdout"]) & (pf.flag_inserted == 0) & (~pf.flag_exclude)]
    hm = summary[(summary.fold == "holdout") & (summary.outliers == "included")].set_index("model")
    overall = cov[(cov.by == "overall")].set_index("interval")
    info = {
        "holdout_weeks": [str(week_map.week_end_date.iloc[origins["holdout"] + 1].date()), str(week_map.week_end_date.iloc[-1].date())],
        "holdout_observed_rows": int(len(win)),
        "holdout_scored_global": int(len(holdout)),
        "holdout_scored_cold_start": int(len(cold_h)),
        "holdout_metrics": {m: {k: round(float(hm.loc[m, k]), 4) for k in ("wmape", "rmse", "bias")} for m in hm.index},
        "interval_nominal": 1 - cfg["intervals"]["alpha"],
        "interval_coverage": {k: round(float(overall.loc[k, "coverage"]), 4) for k in overall.index},
        "interval_mean_width": {k: round(float(overall.loc[k, "mean_width"]), 2) for k in overall.index},
        "conformal_margin_units": round(qhat, 4),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "run_summary.json").write_text(json.dumps(info, indent=2))
    log.info("Done in %.0fs. Holdout WMAPE: %s", time.time() - t0, {m: v["wmape"] for m, v in info["holdout_metrics"].items()})


if __name__ == "__main__":
    main()
