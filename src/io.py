from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    REQUIRED_OPENING_COLUMNS,
    REQUIRED_PURCHASE_COLUMNS,
    REQUIRED_TARGET_COLUMNS,
    REQUIRED_TRANSACTION_COLUMNS,
    SUB_KEL_FALLBACK,
)
from .utils import clean_sku, ensure_columns, normalize_columns, numeric_series, read_csv_bytes


@dataclass
class RawInputs:
    opening: pd.DataFrame
    transactions: pd.DataFrame
    purchases: Optional[pd.DataFrame]
    targets: Optional[pd.DataFrame]


def _ensure_optional_sub_kel(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a consistent product hierarchy when source files have no sub_kel."""
    out = df.copy()
    if "sub_kel" not in out.columns:
        out["sub_kel"] = SUB_KEL_FALLBACK
    else:
        s = out["sub_kel"].fillna("").astype(str).str.strip()
        out["sub_kel"] = s.mask(s.eq(""), SUB_KEL_FALLBACK)
    return out


def _mode_year(series: pd.Series) -> Optional[int]:
    s = pd.to_datetime(series, errors="coerce").dropna()
    if s.empty:
        return None
    years = s.dt.year.value_counts()
    return int(years.index[0]) if len(years) else None


def _candidate_date_from_transaction_code(kd_trx: pd.Series) -> pd.Series:
    """Extract YYMMDD from transaction codes such as THJ2601010001C / OA-2601010001-HJ."""
    code = kd_trx.fillna("").astype(str)
    token = code.str.extract(r"(\d{6})", expand=False)
    return pd.to_datetime(token, format="%y%m%d", errors="coerce")


def _parse_transaction_datetime(df: pd.DataFrame, analysis_year: Optional[int] = None) -> pd.DataFrame:
    """Parse a normal datetime column, with a safe date-only fallback from kd_trx.

    Some POS exports contain `tgl` as MM:SS.0 only (for example 16:44.0), so
    interpreting it with pandas would incorrectly use today's date and may reject
    values where the first component is >23. In that format the calendar date is
    reconstructed from the YYMMDD embedded in kd_trx. The hour is deliberately not
    invented; `time_available=False` is kept so Hourly Sales can be disabled.
    """
    out = df.copy()
    raw = out["tgl"].fillna("").astype(str).str.strip()
    out["tgl_raw"] = raw

    # Only parse values that actually contain a calendar date. This prevents
    # strings such as 16:44.0 from being interpreted as "today at 16:44".
    has_calendar_date = raw.str.contains(
        r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        regex=True,
        na=False,
    )
    parsed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    if has_calendar_date.any():
        parsed.loc[has_calendar_date] = pd.to_datetime(raw.loc[has_calendar_date], errors="coerce")

    code_date = _candidate_date_from_transaction_code(out["kd_trx"])

    if analysis_year is None:
        parsed_year = _mode_year(parsed)
        code_year = _mode_year(code_date)
        analysis_year = parsed_year or code_year

    source = pd.Series("UNRESOLVED", index=out.index, dtype=object)
    source.loc[parsed.notna()] = "TGL"

    need_fallback = parsed.isna() & code_date.notna()
    if analysis_year is not None:
        in_year = code_date.dt.year.eq(int(analysis_year))
        use_code = need_fallback & in_year
        outside = need_fallback & ~in_year
        parsed.loc[use_code] = code_date.loc[use_code]
        source.loc[use_code] = "KD_TRX_DATE"
        source.loc[outside] = "OUTSIDE_ANALYSIS_YEAR"
    else:
        parsed.loc[need_fallback] = code_date.loc[need_fallback]
        source.loc[need_fallback] = "KD_TRX_DATE"

    # A reliable hour exists only when the original source had a full date + time.
    has_clock = raw.str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)
    out["time_available"] = parsed.notna() & has_calendar_date & has_clock
    out["date_parse_status"] = source
    out["tgl"] = parsed
    return out


def load_opening_bytes(data: bytes) -> pd.DataFrame:
    df = normalize_columns(read_csv_bytes(data, dtype={"sku": str}))
    ensure_columns(df, REQUIRED_OPENING_COLUMNS, "Stock Awal")
    df = _ensure_optional_sub_kel(df)
    df["sku"] = clean_sku(df["sku"])
    for c in ["saldo_awal", "hrg_beli", "subtotal"]:
        df[c] = numeric_series(df[c]).fillna(0.0)
    for c in ["nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def load_transactions_bytes(data: bytes, analysis_year: Optional[int] = None) -> pd.DataFrame:
    df = normalize_columns(read_csv_bytes(data, dtype={"sku": str, "kd_trx": str, "tgl": str}))
    ensure_columns(df, REQUIRED_TRANSACTION_COLUMNS, "Kartu Stok")
    df = _ensure_optional_sub_kel(df)
    df["sku"] = clean_sku(df["sku"])
    df["kd_trx"] = df["kd_trx"].fillna("").astype(str).str.strip()
    df = _parse_transaction_datetime(df, analysis_year=analysis_year)
    for c in ["stock_in", "stock_out", "harga", "subtotal"]:
        df[c] = numeric_series(df[c]).fillna(0.0)
    for c in ["nama_barang", "supplier", "subdept", "kel_barang", "sub_kel", "keterangan"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def load_purchase_bytes(data: bytes) -> pd.DataFrame:
    df = normalize_columns(read_csv_bytes(data, dtype={"sku": str, "no_faktur_beli": str}))
    if "hrg_beli" in df.columns and "harga_beli" not in df.columns:
        df = df.rename(columns={"hrg_beli": "harga_beli"})
    ensure_columns(df, REQUIRED_PURCHASE_COLUMNS, "Histori Pembelian")
    df["sku"] = clean_sku(df["sku"])
    df["tgl"] = pd.to_datetime(df["tgl"], errors="coerce")
    df["harga_beli"] = numeric_series(df["harga_beli"])
    df["no_faktur_beli"] = df["no_faktur_beli"].fillna("").astype(str).str.strip()
    return df


def load_target_bytes(data: bytes, default_location: str = "IDK-ATP") -> pd.DataFrame:
    df = normalize_columns(read_csv_bytes(data))
    if "target" in df.columns and "target_omzet" not in df.columns:
        df = df.rename(columns={"target": "target_omzet"})
    ensure_columns(df, REQUIRED_TARGET_COLUMNS, "Target Cabang")
    df["bulan"] = df["bulan"].astype(str).str.strip()
    df["bulan"] = pd.to_datetime(df["bulan"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["target_omzet"] = numeric_series(df["target_omzet"])
    if "lokasi" not in df.columns:
        df["lokasi"] = default_location
    else:
        df["lokasi"] = df["lokasi"].fillna(default_location).astype(str).str.strip()
    for c in ["target_transaksi", "target_gross_profit", "target_margin_pct"]:
        if c in df.columns:
            df[c] = numeric_series(df[c])
    return df


def _infer_analysis_year(
    purchases: Optional[pd.DataFrame],
    targets: Optional[pd.DataFrame],
) -> Optional[int]:
    # Target is the strongest indicator because each application run represents
    # one branch/year. Purchase history is the next-best source.
    if targets is not None and not targets.empty and targets["bulan"].notna().any():
        return _mode_year(targets["bulan"])
    if purchases is not None and not purchases.empty and purchases["tgl"].notna().any():
        return _mode_year(purchases["tgl"])
    return None


def load_raw_inputs(
    opening_bytes: bytes,
    transaction_bytes: bytes,
    purchase_bytes: Optional[bytes] = None,
    target_bytes: Optional[bytes] = None,
    default_location: str = "IDK-ATP",
) -> RawInputs:
    purchases = load_purchase_bytes(purchase_bytes) if purchase_bytes else None
    targets = load_target_bytes(target_bytes, default_location=default_location) if target_bytes else None
    analysis_year = _infer_analysis_year(purchases, targets)
    return RawInputs(
        opening=load_opening_bytes(opening_bytes),
        transactions=load_transactions_bytes(transaction_bytes, analysis_year=analysis_year),
        purchases=purchases,
        targets=targets,
    )
