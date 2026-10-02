"""Build the contiguous weekly store-UPC panel and handle missing weeks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("panel")

SALES_COLS = ["units", "visits", "hhs", "spend"]
PROMO_COLS = ["feature", "display", "tpr_only"]


def promo_type(df: pd.DataFrame) -> pd.Series:
    """0 none, 1 TPR only, 2 display only, 3 feature only, 4 feature + display."""
    return np.select(
        [(df.feature == 1) & (df.display == 1), df.feature == 1, df.display == 1, df.tpr_only == 1],
        [4, 3, 2, 1],
        default=0,
    )


def build_panel(tx: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, week_map).

    Each store-UPC series is made contiguous between its first and last observed week.
    Weeks missing inside that span are inserted with zero sales and flag_inserted = 1.
    Weeks before the first or after the last observation are not created (not launched / discontinued).
    """
    tx = tx.copy()
    weeks = np.sort(tx["week_end_date"].unique())
    week_map = pd.DataFrame({"week_idx": np.arange(len(weeks)), "week_end_date": pd.to_datetime(weeks)})
    tx["week_idx"] = tx["week_end_date"].map(dict(zip(week_map.week_end_date, week_map.week_idx))).astype(int)
    tx["series_id"] = tx.groupby(["store_num", "upc"]).ngroup()

    span = tx.groupby("series_id")["week_idx"].agg(["min", "max"])
    lengths = (span["max"] - span["min"] + 1).to_numpy()
    sid = np.repeat(span.index.to_numpy(), lengths)
    offs = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame({"series_id": sid, "week_idx": np.repeat(span["min"].to_numpy(), lengths) + offs})

    keep = ["series_id", "week_idx", "store_num", "upc", *SALES_COLS, "price", "base_price", *PROMO_COLS,
            "flag_outlier", "flag_exclude", "flag_visits_gt_units"]
    panel = grid.merge(tx[keep], on=["series_id", "week_idx"], how="left", indicator=True)
    panel["flag_inserted"] = (panel.pop("_merge") == "left_only").astype(int)

    ids = tx.drop_duplicates("series_id").set_index("series_id")[["store_num", "upc"]]
    miss = panel["store_num"].isna()
    panel.loc[miss, "store_num"] = panel.loc[miss, "series_id"].map(ids["store_num"])
    panel.loc[miss, "upc"] = panel.loc[miss, "series_id"].map(ids["upc"])
    panel[["store_num", "upc"]] = panel[["store_num", "upc"]].astype(int)

    ins = panel["flag_inserted"] == 1
    panel.loc[ins, SALES_COLS] = 0
    panel.loc[ins, PROMO_COLS] = 0
    panel[["flag_outlier", "flag_exclude", "flag_visits_gt_units"]] = (
        panel[["flag_outlier", "flag_exclude", "flag_visits_gt_units"]].fillna(False).astype(bool)
    )
    panel = panel.sort_values(["series_id", "week_idx"]).reset_index(drop=True)
    for c in ["price", "base_price"]:
        panel[c] = panel.groupby("series_id")[c].transform(lambda s: s.ffill().bfill())
    panel[SALES_COLS] = panel[SALES_COLS].astype(float)
    panel[PROMO_COLS] = panel[PROMO_COLS].astype(int)

    panel["discount_pct"] = (1 - panel["price"] / panel["base_price"]).clip(lower=0, upper=1)
    panel["promo_type"] = promo_type(panel)
    panel = panel.merge(week_map, on="week_idx", how="left")
    panel["week_end_date"] = pd.to_datetime(panel["week_end_date"])

    if not cfg["data"]["insert_missing_as_zero"]:
        panel = panel[panel["flag_inserted"] == 0].reset_index(drop=True)

    log.info("Panel: %d rows, %d series, %d inserted zero rows (%.1f%%)",
             len(panel), panel.series_id.nunique(), int(panel.flag_inserted.sum()), 100 * panel.flag_inserted.mean())
    return panel, week_map


if __name__ == "__main__":
    from .data_loading import load_tables
    from .quality_checks import run
    from .utils import load_config

    cfg = load_config()
    clean, _ = run(cfg, load_tables(cfg))
    p, _ = build_panel(clean, cfg)
    print(p.head())
