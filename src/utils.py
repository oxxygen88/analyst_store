from __future__ import annotations

import hashlib
import io
import math
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def normalize_colname(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_colname(c) for c in out.columns]
    return out


def numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"(?i)rp", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_sku(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def read_csv_bytes(data: bytes, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), low_memory=False, **kwargs)


def file_fingerprint(*payloads: Optional[bytes]) -> str:
    h = hashlib.sha256()
    for payload in payloads:
        if payload is None:
            h.update(b"<NONE>")
        else:
            h.update(payload)
    return h.hexdigest()[:20]


def safe_div(numerator, denominator, default=np.nan):
    if isinstance(denominator, pd.Series):
        den = denominator.replace(0, np.nan)
        result = numerator / den
        return result.fillna(default) if not (isinstance(default, float) and math.isnan(default)) else result
    if denominator in (0, None) or (isinstance(denominator, float) and np.isnan(denominator)):
        return default
    return numerator / denominator


def rupiah(value, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if digits == 0:
        return f"{sign}Rp. {value:,.0f}".replace(",", ".")
    return f"{sign}Rp. {value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value * 100:.{digits}f}%"


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.to_period("M").to_timestamp()


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.to_period("M").to_timestamp("M")


def period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{pd.Timestamp(start):%d %b %Y} – {pd.Timestamp(end):%d %b %Y}"


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    if series.empty:
        return series.astype(float)
    # pct rank: high value gets high score by default.
    rank = series.rank(method="average", pct=True, ascending=ascending)
    return rank * 100.0


def ensure_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name}: kolom wajib tidak ditemukan: {', '.join(missing)}")


_CURRENCY_EXACT = {
    "revenue", "net_sales", "gross_sales", "sales_return", "target", "target_omzet", "gap",
    "projected_gap", "projected_month_end", "required_daily_sales", "atv", "hpp", "net_hpp",
    "gross_profit", "inventory_value", "current_stock_value", "current_cost", "harga", "harga_beli",
    "subtotal", "hpp_unit", "estimated_potential_revenue", "stock_value", "purchase_value",
    "transfer_in_value", "transfer_out_value", "actual", "base_projection", "scenario_projection",
}
_PERCENT_EXACT = {
    "revenue_share", "cumulative_share", "gross_margin", "inventory_share", "share_lines",
    "achievement", "pace_achievement", "growth_30d",
}

def style_dataframe(df: pd.DataFrame):
    """Pandas Styler untuk tampilan Streamlit: Rupiah memakai Rp. + separator ribuan Indonesia."""
    formatters = {}
    for col in df.columns:
        name = str(col).lower()
        if name in _CURRENCY_EXACT or name.endswith("_revenue") or name.endswith("_value") or name.endswith("_sales") or name.endswith("_profit") or name.endswith("_hpp"):
            if pd.api.types.is_numeric_dtype(df[col]):
                formatters[col] = rupiah
        elif name in _PERCENT_EXACT or name.endswith("_share") or name.endswith("_margin"):
            if pd.api.types.is_numeric_dtype(df[col]):
                formatters[col] = pct
    return df.style.format(formatters, na_rep="-")
