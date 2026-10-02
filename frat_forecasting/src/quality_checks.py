"""Data-quality checks, flags and price cleaning for the transaction table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger, resolve

log = get_logger("quality_checks")

SERIES = ["store_num", "upc"]


def add_flags(tx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add boolean flags for suspicious rows. Rows are never deleted here."""
    d = cfg["data"]
    tx = tx.copy()
    upv = tx["units"] / tx["visits"].replace(0, np.nan)
    vph = tx["visits"] / tx["hhs"].replace(0, np.nan)
    tx["flag_visits_gt_units"] = tx["visits"] > tx["units"]
    tx["flag_high_units_per_visit"] = upv > d["outlier_units_per_visit"]
    tx["flag_high_visits_per_hh"] = vph > d["outlier_visits_per_hh"]
    tx["flag_zero_units_with_visits"] = (tx["units"] <= 0) & (tx["visits"] > 0)
    # statistical outlier flag used for the "with / without outliers" comparison
    tx["flag_outlier"] = tx["flag_high_units_per_visit"] | tx["flag_high_visits_per_hh"]
    tx["flag_price_invalid"] = tx["price"].isna() | (tx["price"] <= 0)
    tx["flag_base_price_invalid"] = tx["base_price"].isna() | (tx["base_price"] <= 0)
    return tx


def clean_prices(tx: pd.DataFrame) -> pd.DataFrame:
    """Refill missing or zero prices from neighbouring weeks of the same store-UPC series."""
    tx = tx.sort_values(SERIES + ["week_end_date"]).reset_index(drop=True)
    for col in ["price", "base_price"]:
        bad = tx[col].isna() | (tx[col] <= 0)
        tx.loc[bad, col] = np.nan
        tx[col] = tx.groupby(SERIES)[col].transform(lambda s: s.ffill().bfill())
    tx["price_imputed"] = tx["flag_price_invalid"] | tx["flag_base_price_invalid"]
    # a row is excluded from training/evaluation if units are untrustworthy or the price is still missing
    tx["flag_exclude"] = tx["flag_zero_units_with_visits"] | tx["price"].isna() | tx["base_price"].isna()
    return tx


def quality_report(tx_raw: pd.DataFrame, tx: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Summary table of issues found (computed on the raw data, before cleaning)."""
    n = len(tx_raw)
    promo_any = (tx_raw["feature"] + tx_raw["display"] + tx_raw["tpr_only"]) > 0
    disc = (tx_raw["price"] / tx_raw["base_price"] - 1)
    n_weeks = tx_raw["week_end_date"].nunique()
    n_series = tx_raw.groupby(SERIES).ngroups
    expected = n_weeks * tx_raw["store_num"].nunique() * tx_raw["upc"].nunique()
    rows = [
        ("Rows", n, ""),
        ("Weeks", n_weeks, f"{tx_raw.week_end_date.min().date()} to {tx_raw.week_end_date.max().date()}"),
        ("Stores in transactions", tx_raw["store_num"].nunique(), ""),
        ("UPCs in transactions", tx_raw["upc"].nunique(), f"{products['upc'].nunique()} in product lookup"),
        ("Store-UPC series", n_series, ""),
        ("Duplicate week-store-UPC rows", int(tx_raw.duplicated(["week_end_date", "store_num", "upc"]).sum()), ""),
        ("UPCs missing from product lookup", len(set(tx_raw.upc) - set(products.upc)), ""),
        ("Stores missing from store lookup", len(set(tx_raw.store_num) - set(stores.store_num)), ""),
        ("Stores with conflicting tier labels", len(stores.attrs.get("tier_conflict_stores", [])),
         "first label kept: " + ", ".join(map(str, stores.attrs.get("tier_conflict_stores", [])))),
        ("Missing price", int(tx_raw["price"].isna().sum()), "refilled from the same series"),
        ("Missing base price", int(tx_raw["base_price"].isna().sum()), "refilled from the same series"),
        ("Price = 0", int((tx_raw["price"] == 0).sum()), "treated as invalid, refilled"),
        ("Units = 0 with visits > 0", int(((tx_raw.units <= 0) & (tx_raw.visits > 0)).sum()), "excluded from training and evaluation"),
        ("Visits > units (impossible)", int((tx_raw.visits > tx_raw.units).sum()), "flagged only"),
        ("Units per visit above threshold", int(tx["flag_high_units_per_visit"].sum()), "flagged outlier, kept"),
        ("Visits per household above threshold", int(tx["flag_high_visits_per_hh"].sum()), "flagged outlier, kept"),
        ("Price above base price", int((tx_raw.price > tx_raw.base_price).sum()), "kept; discount depth floored at 0"),
        ("Price more than 10% above base price", int((disc > 0.10).sum()), ""),
        ("Promo flag set but no price cut", int((promo_any & (tx_raw.price >= tx_raw.base_price)).sum()), "expected for feature/display"),
        ("Price cut over 5% with no promo flag", int(((disc < -0.05) & ~promo_any).sum()), ""),
        ("Rows in a full panel", int(expected), f"actual rows are {n / expected:.0%} of this"),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "note"])


def run(cfg: dict, tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag and clean the transactions; write the summary to outputs/. Returns (clean_tx, report)."""
    tx_raw = tables["transactions"]
    flagged = add_flags(tx_raw, cfg)
    report = quality_report(tx_raw, flagged, tables["products"], tables["stores"])
    clean = clean_prices(flagged)
    out = resolve(cfg, "output_dir")
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "data_quality_summary.csv", index=False)
    log.info("Data-quality summary written (%d checks); %d rows excluded", len(report), int(clean["flag_exclude"].sum()))
    return clean, report
