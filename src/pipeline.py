from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .analytics import anomaly_tables, hpp_coverage, inventory_health
from .hpp import resolve_commercial_hpp
from .io import RawInputs
from .transform import build_master, daily_movement, enrich_transactions


@dataclass
class AnalysisBundle:
    opening: pd.DataFrame
    tx: pd.DataFrame
    purchases: Optional[pd.DataFrame]
    targets: Optional[pd.DataFrame]
    master: pd.DataFrame
    daily: pd.DataFrame
    min_date: pd.Timestamp
    max_date: pd.Timestamp


def build_bundle(raw: RawInputs) -> AnalysisBundle:
    tx = enrich_transactions(raw.transactions)
    tx = resolve_commercial_hpp(tx, raw.opening, raw.purchases)
    master = build_master(raw.opening, tx)
    daily = daily_movement(tx)
    min_date = tx["date"].dropna().min()
    max_date = tx["date"].dropna().max()
    return AnalysisBundle(
        opening=raw.opening,
        tx=tx,
        purchases=raw.purchases,
        targets=raw.targets,
        master=master,
        daily=daily,
        min_date=pd.Timestamp(min_date),
        max_date=pd.Timestamp(max_date),
    )
