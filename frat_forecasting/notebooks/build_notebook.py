"""Generates notebooks/01_eda_data_quality.ipynb (run once; execute the notebook in Jupyter afterwards)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# EDA and data-quality report: Breakfast at the Frat

Weekly product-store sales for four categories (bag snacks, cold cereal, frozen pizza, oral hygiene).
This notebook documents what was checked, what was found and how each issue is handled in the modelling pipeline.
Run it from the `notebooks/` folder after placing the workbook in `data/raw/` (see the README)."""))

cells.append(code("""import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

from src.utils import load_config
from src.data_loading import load_tables, prepare_lookups, merge_tables
from src.quality_checks import run as run_quality
from src.panel import build_panel
from src.features import build_panel_features
from src.elasticity import promo_lift

cfg = load_config()
tables = load_tables(cfg)
products, stores = prepare_lookups(tables["products"], tables["stores"])
tables["products"], tables["stores"] = products, stores
tx = tables["transactions"]
print({k: v.shape for k, v in tables.items()})"""))

cells.append(md("## 1. Structure and joins"))
cells.append(code("""print("Weeks:", tx.week_end_date.nunique(), tx.week_end_date.min().date(), "to", tx.week_end_date.max().date())
print("Week spacing (days):", tx.week_end_date.drop_duplicates().sort_values().diff().dt.days.value_counts().to_dict())
print("Stores:", tx.store_num.nunique(), "| UPCs in transactions:", tx.upc.nunique(), "| UPCs in lookup:", products.upc.nunique())
print("Stores with conflicting tier labels in the lookup:", stores.attrs.get("tier_conflict_stores"))
display(products.groupby(["category", "sub_category"]).size().rename("upcs").reset_index())
display(stores.seg_value_name.value_counts().rename("stores").to_frame().T)
display(stores.address_state_prov_code.value_counts().rename("stores").to_frame().T)"""))

cells.append(md("## 2. Data-quality summary"))
cells.append(code("""clean, report = run_quality(cfg, tables)
display(report)"""))

cells.append(md("""## 3. Outliers

The user guide suggests units per visit and visits per household as outlier screens. Flagged rows are kept in
the data and reported separately, because they reflect real behaviour."""))
cells.append(code("""d = clean.assign(upv=clean.units / clean.visits.replace(0, np.nan), vph=clean.visits / clean.hhs.replace(0, np.nan))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
d.upv.clip(upper=15).hist(bins=60, ax=ax[0]); ax[0].set_title("Units per visit (clipped at 15)"); ax[0].axvline(cfg["data"]["outlier_units_per_visit"], color="r")
d.vph.clip(upper=10).hist(bins=60, ax=ax[1]); ax[1].set_title("Visits per household (clipped at 10)"); ax[1].axvline(cfg["data"]["outlier_visits_per_hh"], color="r")
plt.tight_layout(); plt.show()
print("Rows flagged as outliers:", int(clean.flag_outlier.sum()), f"({clean.flag_outlier.mean():.2%})")
display(clean.nlargest(5, "units")[["week_end_date", "store_num", "upc", "units", "visits", "hhs", "price"]])"""))

cells.append(md("""## 4. Panel completeness

A full panel would have one row per week for every store-UPC pair. The raw data is sparser, so the pipeline makes
each series contiguous between its first and last observed week and inserts missing weeks inside that span as zero
units (flagged `flag_inserted`). Weeks before the first or after the last observation are not created."""))
cells.append(code("""panel, week_map = build_panel(clean, cfg)
g = panel.groupby("series_id").agg(first=("week_idx", "min"), last=("week_idx", "max"), obs=("flag_inserted", lambda s: (s == 0).sum()), ins=("flag_inserted", "sum"))
n_weeks = panel.week_idx.max() + 1
summary = pd.Series({
    "series": len(g),
    "full 156 weeks observed": int((g.obs == n_weeks).sum()),
    "start after week 1": int((g["first"] > 0).sum()),
    "end before last week": int((g["last"] < n_weeks - 1).sum()),
    "series with interior gaps": int((g.ins > 0).sum()),
    "inserted zero rows": int(g.ins.sum()),
})
display(summary.to_frame("count"))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
g.obs.hist(bins=40, ax=ax[0]); ax[0].set_title("Observed weeks per series")
sample = panel[panel.series_id.isin(g.sample(150, random_state=1).index)].pivot(index="series_id", columns="week_idx", values="flag_inserted")
full = pd.DataFrame(np.nan, index=sample.index, columns=range(n_weeks)); full.update(sample)
ax[1].imshow(full.values, aspect="auto", cmap="coolwarm", interpolation="nearest"); ax[1].set_title("150 random series: blue = observed, red = inserted, white = outside span")
ax[1].set_xlabel("week index"); plt.tight_layout(); plt.show()"""))

cells.append(md("## 5. Demand over time"))
cells.append(code("""pf = build_panel_features(panel, products, stores)
obs = pf[pf.flag_inserted == 0].assign(category=lambda x: x.category.astype(str))
t = obs.groupby(["week_end_date", "category"]).units.sum().unstack()
t.plot(figsize=(10, 3.6), lw=1.1, title="Weekly units by category"); plt.show()
hol = obs.groupby("week_end_date").agg(units=("units", "sum"), n=("units", "size"), thanks=("thanksgiving_week", "max"), xmas=("christmas_week", "max"), sb=("superbowl_week", "max"))
hol["units_per_row"] = hol.units / hol.n
print("Mean weekly units per observed row, by calendar event week:")
print(hol.groupby(["thanks", "xmas", "sb"]).units_per_row.mean().round(2))"""))

cells.append(md("## 6. Prices"))
cells.append(code("""ratio = (obs.price / obs.base_price)
print("Price range by category (min / median / max shelf price):")
display(obs.groupby("category").price.describe()[["min", "50%", "max"]].round(2))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ratio.clip(0.4, 1.4).hist(bins=80, ax=ax[0]); ax[0].set_title("Price / base price")
smp = obs.sample(20000, random_state=0)
for c, gg in smp.groupby("category"):
    ax[1].scatter(gg.price / gg.base_price, np.log1p(gg.units / gg.o_mean52.replace(0, np.nan)), s=2, alpha=0.2, label=c)
ax[1].set_xlim(0.4, 1.3); ax[1].set_title("Units index vs price ratio (sample)"); ax[1].set_xlabel("price / base price"); ax[1].legend(markerscale=4, fontsize=7)
plt.tight_layout(); plt.show()"""))

cells.append(md("""## 7. Promotions

The three promotion flags never overlap except feature + display, so promotions are encoded as five states:
none, TPR only, display only, feature only, feature + display. Lift is measured against each series' own
non-promoted weeks (median ratio), which controls for store and product size."""))
cells.append(code("""print(obs.assign(promo=obs.promo_type.map({0:"none",1:"TPR only",2:"display only",3:"feature only",4:"feature + display"})).promo.value_counts(normalize=True).round(3))
pl = promo_lift(pf)
display(pl[pl.split == "overall"].round(3))
display(pl[pl.split == "category"].pivot(index="segment", columns="promo", values="units_lift").round(2))"""))

cells.append(md("## 8. Store segments"))
cells.append(code("""o = obs.assign(tier=obs.tier.astype(str), state=obs.state.astype(str))
display(o.groupby("tier").agg(stores=("store_num", "nunique"), avg_units_per_row=("units", "mean"), avg_price=("price", "mean"), promo_share=("promo_type", lambda s: (s > 0).mean())).round(2))
display(o.groupby("state").agg(stores=("store_num", "nunique"), avg_units_per_row=("units", "mean"), avg_price=("price", "mean")).round(2))
store_tot = o.groupby("store_num").agg(units=("units", "sum"), baskets=("avg_weekly_baskets", "first"))
print("Correlation between store total units and average weekly baskets:", round(store_tot.corr().iloc[0, 1], 3))"""))

cells.append(md("""## 9. Decisions carried into the pipeline

- Rows are flagged, not deleted; rows with zero units but positive visits are excluded from training and scoring.
- Missing or zero prices are refilled from neighbouring weeks of the same store-UPC series.
- Two stores have conflicting price-tier labels in the lookup; the first label is used.
- Missing weeks inside a series' active span are inserted as zero units and flagged; they are never scored.
- Outliers are kept, and headline metrics are reported with and without them.
- Promotions are modelled as five mutually exclusive states."""))

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
from pathlib import Path
nbf.write(nb, Path(__file__).with_name("01_eda_data_quality.ipynb"))
print("wrote notebook with", len(cells), "cells")
