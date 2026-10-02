"""Feature engineering.

Design (direct multi-horizon, leakage-safe)
-------------------------------------------
Every training row is a (series, origin, horizon h) triple that predicts the target week t = origin + h.
Features come from two clearly separated sources:

* "origin" features (prefix ``o_``) summarise history up to and including the origin week only;
* "known-in-advance" features describe the target week itself: price, promotion, calendar and static
  product/store attributes. Using the target week's price and promotion is an explicit assumption
  (retail promotions are scheduled before they run) and is stated in the README.

Lags of the target week (lag52) are always at least 39 weeks before it, so they are available at the origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar

from .utils import get_logger

log = get_logger("features")

CAT_COLS = ["upc", "category", "sub_category", "manufacturer", "tier", "state"]

O_COLS = [
    "o_units_last", "o_mean4", "o_mean8", "o_mean13", "o_mean26", "o_mean52", "o_std13",
    "o_nonpromo13", "o_promo_rate13", "o_disc13", "o_zero_rate13", "o_price", "o_base", "o_hist_len",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "history": ["h", *O_COLS, "lag52", "lag52_mean3"],
    "price": ["price", "base_price", "discount_pct", "price_gap_cat"],
    "promo": ["feature", "display", "tpr_only", "promo_type"],
    "product": ["upc", "category", "sub_category", "manufacturer", "size_value"],
    "store": ["tier", "state", "sales_area_size_num", "avg_weekly_baskets"],
    "calendar": ["week_of_year", "month", "holiday_week", "easter_week", "thanksgiving_week",
                 "christmas_week", "superbowl_week"],
}
ALL_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

SUPERBOWL_SUNDAYS = pd.to_datetime(["2009-02-01", "2010-02-07", "2011-02-06", "2012-02-05"])


# ---------------------------------------------------------------- static attributes
def add_static(panel: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    prod = products[["upc", "category", "sub_category", "manufacturer", "size_value"]]
    st = stores[["store_num", "address_state_prov_code", "sales_area_size_num", "avg_weekly_baskets", "seg_value_name"]]
    st = st.rename(columns={"address_state_prov_code": "state", "seg_value_name": "tier"})
    out = panel.merge(prod, on="upc", how="left").merge(st, on="store_num", how="left")
    for c in CAT_COLS:
        out[c] = out[c].astype(str).astype("category")
    return out


# ---------------------------------------------------------------- calendar
def _in_week(week_end: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """True if any of `dates` falls in the 7 days ending on each week_end date."""
    we = week_end.to_numpy("datetime64[D]")
    d = np.sort(dates.to_numpy("datetime64[D]"))
    left = np.searchsorted(d, we - np.timedelta64(6, "D"), side="left")
    right = np.searchsorted(d, we, side="right")
    return (right - left) > 0


def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    we = out["week_end_date"]
    yrs = range(we.dt.year.min() - 1, we.dt.year.max() + 2)
    fed = USFederalHolidayCalendar().holidays(f"{min(yrs)}-01-01", f"{max(yrs)}-12-31")
    thanks = pd.DatetimeIndex([d for d in fed if d.month == 11 and d.dayofweek == 3])
    xmas = pd.DatetimeIndex([pd.Timestamp(y, 12, 25) for y in yrs])
    east = pd.DatetimeIndex([pd.Timestamp(easter(y)) for y in yrs])
    out["week_of_year"] = we.dt.isocalendar().week.astype(int)
    out["month"] = we.dt.month
    out["holiday_week"] = _in_week(we, fed).astype(int)
    out["easter_week"] = _in_week(we, east).astype(int)
    out["thanksgiving_week"] = _in_week(we, thanks).astype(int)
    out["christmas_week"] = _in_week(we, xmas).astype(int)
    out["superbowl_week"] = _in_week(we, SUPERBOWL_SUNDAYS).astype(int)
    return out


# ---------------------------------------------------------------- price and history
def add_price_gap(panel: pd.DataFrame) -> pd.DataFrame:
    """Price relative to the average price in the same category, store and week."""
    out = panel.copy()
    mean_p = out.groupby(["store_num", "week_idx", "category"], observed=True)["price"].transform("mean")
    out["price_gap_cat"] = out["price"] / mean_p
    return out


def _roll(panel: pd.DataFrame, col: str, window: int, fn: str = "mean", min_periods: int = 1) -> pd.Series:
    r = panel.groupby("series_id")[col].rolling(window, min_periods=min_periods)
    return getattr(r, fn)().reset_index(level=0, drop=True).reindex(panel.index)


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """As-of features: the value in row (series, week w) uses data up to and including week w."""
    out = panel.sort_values(["series_id", "week_idx"]).reset_index(drop=True)
    g = out.groupby("series_id")
    out["o_units_last"] = out["units"]
    for w in (4, 8, 13, 26, 52):
        out[f"o_mean{w}"] = _roll(out, "units", w)
    out["o_std13"] = _roll(out, "units", 13, "std", min_periods=2)
    out["_np"] = out["units"].where(out["promo_type"] == 0)
    out["o_nonpromo13"] = _roll(out, "_np", 13)
    out["_promo"] = (out["promo_type"] > 0).astype(float)
    out["o_promo_rate13"] = _roll(out, "_promo", 13)
    out["o_disc13"] = _roll(out, "discount_pct", 13)
    out["_zero"] = (out["units"] == 0).astype(float)
    out["o_zero_rate13"] = _roll(out, "_zero", 13)
    out["o_price"] = out["price"]
    out["o_base"] = out["base_price"]
    out["o_hist_len"] = g.cumcount() + 1
    # seasonal lags relative to the row's own week (rows are contiguous weeks within a series)
    s = [g["units"].shift(k) for k in (51, 52, 53)]
    out["lag52"] = s[1]
    out["lag52_mean3"] = pd.concat(s, axis=1).mean(axis=1)
    return out.drop(columns=["_np", "_promo", "_zero"])


def build_panel_features(panel: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    out = add_static(panel, products, stores)
    out = add_calendar(out)
    out = add_price_gap(out)
    out = add_history_features(out)
    log.info("Panel features built: %d rows x %d columns", *out.shape)
    return out


# ---------------------------------------------------------------- supervised rows
TARGET_SIDE = [
    "series_id", "week_idx", "week_end_date", "store_num", "upc", "units", "spend", "flag_inserted", "flag_exclude",
    "flag_outlier", "lag52", "lag52_mean3",
    *FEATURE_GROUPS["price"], *FEATURE_GROUPS["promo"], *FEATURE_GROUPS["product"],
    *FEATURE_GROUPS["store"], *FEATURE_GROUPS["calendar"],
]
TARGET_SIDE = list(dict.fromkeys(TARGET_SIDE))


def build_supervised(panel: pd.DataFrame, origins: list[int], horizon: int) -> pd.DataFrame:
    """Create (series, origin, h) rows for the given origins."""
    origins = sorted(set(int(o) for o in origins))
    otbl = panel.loc[panel["week_idx"].isin(origins), ["series_id", "week_idx", *O_COLS]].rename(columns={"week_idx": "origin"})
    frames = []
    for h in range(1, horizon + 1):
        mask = (panel["week_idx"] - h).isin(origins)
        t = panel.loc[mask, TARGET_SIDE].copy()
        t["origin"] = t["week_idx"] - h
        t["h"] = h
        frames.append(t.merge(otbl, on=["series_id", "origin"], how="inner"))
    sup = pd.concat(frames, ignore_index=True).rename(columns={"week_idx": "target_week", "units": "y"})
    num = sup.select_dtypes("float64").columns
    sup[num] = sup[num].astype("float32")
    sup["history_bucket"] = pd.cut(sup["o_hist_len"], [0, 12, 51, 10_000], labels=["<13 wks", "13-51 wks", "52+ wks"])
    log.info("Supervised rows: %d (origins=%d, horizons=1..%d)", len(sup), len(origins), horizon)
    return sup


def make_origins(last_week_idx: int, horizon: int, stride: int, min_origin: int, n_cv_folds: int) -> dict:
    """Origins for the holdout, the CV folds and the training grid.

    Holdout origin = last_week_idx - horizon (the last training week); its targets are the final `horizon` weeks.
    CV origins step back by `horizon` weeks, so fold test windows do not overlap.
    """
    holdout = int(last_week_idx) - horizon
    cv = [holdout - horizon * k for k in range(1, n_cv_folds + 1)]
    grid = list(range(holdout, min_origin - 1, -stride))
    return {"holdout": holdout, "cv": cv, "all": sorted(set(grid) | set(cv) | {holdout})}


def feature_list(groups: list[str]) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]
