"""Shared helpers: config loading, logging and forecast metrics."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ------------------------------------------------------------------ metrics
def wmape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = y.sum()
    return float(np.abs(y - p).sum() / d) if d > 0 else np.nan


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def bias(y, p) -> float:
    """Signed relative bias: (sum forecast - sum actual) / sum actual."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = y.sum()
    return float((p.sum() - d) / d) if d > 0 else np.nan


def metric_row(y, p) -> dict:
    return {"n": int(len(y)), "wmape": wmape(y, p), "rmse": rmse(y, p), "bias": bias(y, p)}


def segment_metrics(df: pd.DataFrame, by: str | list[str], y: str = "actual", p: str = "forecast") -> pd.DataFrame:
    by = [by] if isinstance(by, str) else by
    rows = []
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **metric_row(g[y], g[p])})
    return pd.DataFrame(rows)
