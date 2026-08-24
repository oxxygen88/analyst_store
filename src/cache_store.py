from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .pipeline import AnalysisBundle


def save_bundle_parquet(bundle: AnalysisBundle, fingerprint: str, root: str = ".cache") -> Optional[Path]:
    """Best-effort disk cache. Requires pyarrow or fastparquet.

    Failure is intentionally non-fatal because Streamlit's in-memory cache remains available.
    """
    path = Path(root) / fingerprint
    path.mkdir(parents=True, exist_ok=True)
    try:
        bundle.opening.to_parquet(path / "opening.parquet", index=False)
        bundle.tx.to_parquet(path / "transactions.parquet", index=False)
        bundle.master.to_parquet(path / "master.parquet", index=False)
        bundle.daily.to_parquet(path / "daily.parquet", index=False)
        if bundle.purchases is not None:
            bundle.purchases.to_parquet(path / "purchases.parquet", index=False)
        if bundle.targets is not None:
            bundle.targets.to_parquet(path / "targets.parquet", index=False)
        meta = {"min_date": str(bundle.min_date), "max_date": str(bundle.max_date)}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return path
    except Exception:
        return None
