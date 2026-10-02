"""Read the dunnhumby workbook, normalise column names and cache as parquet."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import get_logger, resolve

log = get_logger("data_loading")

SHEETS = {
    "transactions": "dh Transaction Data",
    "products": "dh Products Lookup",
    "stores": "dh Store Lookup",
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def read_workbook(xlsx_path: Path, header_row: int = 1) -> dict[str, pd.DataFrame]:
    """Read the three data sheets. The real header sits on row 2 (index 1)."""
    log.info("Reading %s (this takes about a minute)", xlsx_path.name)
    sheets = pd.read_excel(xlsx_path, sheet_name=list(SHEETS.values()), header=header_row, engine="openpyxl")
    out = {k: _clean_columns(sheets[v]) for k, v in SHEETS.items()}
    # the store key is STORE_ID in the lookup but STORE_NUM in transactions
    out["stores"] = out["stores"].rename(columns={"store_id": "store_num"})
    out["transactions"]["week_end_date"] = pd.to_datetime(out["transactions"]["week_end_date"])
    return out


def parse_size(s: str) -> tuple[float, str]:
    """'12.25 OZ' -> (12.25, 'OZ'); unknown formats -> (nan, '')."""
    m = re.match(r"\s*([\d.]+)\s*([A-Za-z]*)", str(s))
    return (float(m.group(1)), m.group(2).upper()) if m else (np.nan, "")


def load_tables(cfg: dict, force: bool = False) -> dict[str, pd.DataFrame]:
    """Load the three tables, using the parquet cache in data/processed when available."""
    proc = resolve(cfg, "processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    cache = {k: proc / f"{k}.parquet" for k in SHEETS}
    if not force and all(p.exists() for p in cache.values()):
        log.info("Loading cached tables from %s", proc)
        return {k: pd.read_parquet(p) for k, p in cache.items()}

    xlsx = resolve(cfg, "raw_xlsx")
    if not xlsx.exists():
        raise FileNotFoundError(f"Place the dunnhumby workbook at {xlsx}")
    tables = read_workbook(xlsx, cfg["data"]["excel_header_row"])
    for k, df in tables.items():
        df.to_parquet(cache[k], index=False)
    return tables


def prepare_lookups(products: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tidy lookup tables: numeric package size and consistent text fields."""
    products = products.copy()
    parsed = products["product_size"].map(parse_size)
    products["size_value"] = [p[0] for p in parsed]
    products["size_unit"] = [p[1] for p in parsed]
    for c in ["category", "sub_category", "manufacturer", "description"]:
        products[c] = products[c].astype(str).str.strip()
    stores = stores.copy()
    stores["seg_value_name"] = stores["seg_value_name"].astype(str).str.strip().str.title()
    # Two stores (4503, 17627) appear twice with conflicting price-tier labels.
    # Neither label is verifiable, so the first occurrence is kept and the conflict is reported.
    conflicts = stores[stores.duplicated("store_num", keep=False)].groupby("store_num")["seg_value_name"].nunique()
    stores.attrs["tier_conflict_stores"] = conflicts[conflicts > 1].index.tolist()
    stores = stores.drop_duplicates("store_num", keep="first").reset_index(drop=True)
    return products, stores


def merge_tables(tx: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Join transactions to product and store attributes."""
    store_cols = ["store_num", "address_state_prov_code", "sales_area_size_num", "avg_weekly_baskets", "seg_value_name"]
    store_cols = [c for c in store_cols if c in stores.columns]
    prod_cols = ["upc", "description", "manufacturer", "category", "sub_category", "size_value"]
    return tx.merge(products[prod_cols], on="upc", how="left").merge(stores[store_cols], on="store_num", how="left")


if __name__ == "__main__":
    from .utils import load_config

    t = load_tables(load_config(), force=True)
    for k, v in t.items():
        print(k, v.shape)
