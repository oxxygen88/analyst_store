from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    REQUIRED_OPENING_COLUMNS,
    REQUIRED_PURCHASE_COLUMNS,
    REQUIRED_TARGET_COLUMNS,
    REQUIRED_TRANSACTION_COLUMNS,
)
from .utils import clean_sku, ensure_columns, normalize_columns, numeric_series, read_csv_bytes


@dataclass
class RawInputs:
    opening: pd.DataFrame
    transactions: pd.DataFrame
    purchases: Optional[pd.DataFrame]
    targets: Optional[pd.DataFrame]


def load_opening_bytes(data: bytes) -> pd.DataFrame:
    df = normalize_columns(read_csv_bytes(data, dtype={"sku": str}))
    ensure_columns(df, REQUIRED_OPENING_COLUMNS, "Stock Awal")
    df["sku"] = clean_sku(df["sku"])
    for c in ["saldo_awal", "hrg_beli", "subtotal"]:
        df[c] = numeric_series(df[c]).fillna(0.0)
    for c in ["nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def load_transactions_bytes(data: bytes) -> pd.DataFrame:
    df = normalize_columns(read_csv_bytes(data, dtype={"sku": str, "kd_trx": str}))
    ensure_columns(df, REQUIRED_TRANSACTION_COLUMNS, "Kartu Stok")
    df["sku"] = clean_sku(df["sku"])
    df["kd_trx"] = df["kd_trx"].fillna("").astype(str).str.strip()
    df["tgl"] = pd.to_datetime(df["tgl"], errors="coerce")
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


def load_raw_inputs(
    opening_bytes: bytes,
    transaction_bytes: bytes,
    purchase_bytes: Optional[bytes] = None,
    target_bytes: Optional[bytes] = None,
    default_location: str = "IDK-ATP",
) -> RawInputs:
    return RawInputs(
        opening=load_opening_bytes(opening_bytes),
        transactions=load_transactions_bytes(transaction_bytes),
        purchases=load_purchase_bytes(purchase_bytes) if purchase_bytes else None,
        targets=load_target_bytes(target_bytes, default_location=default_location) if target_bytes else None,
    )
