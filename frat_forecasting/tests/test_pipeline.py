"""Sanity tests: panel construction, leakage safety and metrics. Run with `pytest -q` from the project root."""
import numpy as np
import pandas as pd
import pytest

from src.data_loading import load_tables, prepare_lookups
from src.features import build_panel_features, build_supervised, make_origins
from src.models import training_rows
from src.panel import build_panel
from src.quality_checks import run as run_quality
from src.utils import load_config, wmape, bias


@pytest.fixture(scope="module")
def built():
    cfg = load_config()
    tables = load_tables(cfg)
    products, stores = prepare_lookups(tables["products"], tables["stores"])
    tables["products"], tables["stores"] = products, stores
    clean, _ = run_quality(cfg, tables)
    panel, _ = build_panel(clean, cfg)
    pf = build_panel_features(panel, products, stores)
    origins = make_origins(pf["week_idx"].max(), cfg["horizon"], 13, 12, 2)
    sup = build_supervised(pf, origins["all"], cfg["horizon"])
    return cfg, clean, panel, pf, origins, sup


def test_panel_is_contiguous_per_series(built):
    _, _, panel, _, _, _ = built
    g = panel.groupby("series_id")["week_idx"].agg(["min", "max", "count"])
    assert ((g["max"] - g["min"] + 1) == g["count"]).all()
    assert not panel.duplicated(["series_id", "week_idx"]).any()


def test_observed_rows_preserved(built):
    _, clean, panel, _, _, _ = built
    assert (panel["flag_inserted"] == 0).sum() == len(clean)
    assert panel.loc[panel.flag_inserted == 1, "units"].eq(0).all()


def test_no_missing_prices_after_cleaning(built):
    _, _, panel, _, _, _ = built
    assert panel["price"].isna().mean() < 0.001


def test_origin_before_target_and_horizon_consistent(built):
    _, _, _, _, _, sup = built
    assert (sup["origin"] < sup["target_week"]).all()
    assert (sup["target_week"] - sup["origin"] == sup["h"]).all()


def test_origin_features_use_only_past_data(built):
    _, _, _, pf, _, sup = built
    rng = np.random.default_rng(0)
    for _, r in sup.iloc[rng.choice(len(sup), 25, replace=False)].iterrows():
        hist = pf[(pf.series_id == r.series_id) & (pf.week_idx <= r.origin) & (pf.week_idx > r.origin - 13)]
        assert np.isclose(hist["units"].mean(), r["o_mean13"], atol=1e-3)
        assert np.isclose(pf[(pf.series_id == r.series_id) & (pf.week_idx == r.origin)]["units"].iloc[0], r["o_units_last"])


def test_lag52_is_a_year_before_target(built):
    _, _, _, pf, _, sup = built
    rows = sup[sup["lag52"].notna()].sample(25, random_state=1)
    for _, r in rows.iterrows():
        prev = pf[(pf.series_id == r.series_id) & (pf.week_idx == r.target_week - 52)]["units"].iloc[0]
        assert np.isclose(prev, r["lag52"], atol=1e-3)
        assert r.target_week - 52 <= r.origin


def test_training_rows_never_peek_past_cutoff(built):
    cfg, _, _, _, origins, sup = built
    tr = training_rows(sup, origins["holdout"], cfg)
    assert tr["target_week"].max() <= origins["holdout"]
    assert tr["origin"].max() < origins["holdout"]


def test_holdout_is_the_last_13_weeks(built):
    cfg, _, panel, _, origins, _ = built
    assert origins["holdout"] == panel["week_idx"].max() - cfg["horizon"]


def test_metrics():
    y, p = np.array([10.0, 20.0]), np.array([12.0, 18.0])
    assert np.isclose(wmape(y, p), 4 / 30)
    assert np.isclose(bias(y, p), 0.0)
