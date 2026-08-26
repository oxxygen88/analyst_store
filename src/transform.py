from __future__ import annotations

import re
import numpy as np
import pandas as pd

from .config import SUB_KEL_FALLBACK


def extract_partner(text: str) -> str:
    text = str(text or "").strip()
    patterns = [
        r"(?i)mutasi\s+ke\s+([^\(]+)",
        r"(?i)mutasi\s+dari\s+([^\(]+)",
        r"(?i)transit\s+online\s+(?:ke|dari)\s+([^\(]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return ""


def extract_reference(text: str) -> str:
    m = re.search(r"\(([^\)]+)\)", str(text or ""))
    return m.group(1).strip() if m else ""


def enrich_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()
    text = df["keterangan"].fillna("").astype(str).str.strip()
    lower = text.str.lower()
    sin, sout = df["stock_in"], df["stock_out"]
    is_in = sin.gt(0) & sout.le(0)
    is_out = sout.gt(0) & sin.le(0)
    is_match = sin.le(0) & sout.le(0)

    df["direction"] = np.select([is_in, is_out, is_match], ["IN", "OUT", "MATCH"], default="BOTH")

    movement = np.full(len(df), "OTHER_MATCH", dtype=object)
    movement[is_in.to_numpy()] = "OTHER_IN"
    movement[is_out.to_numpy()] = "OTHER_OUT"

    exact = lower.eq
    movement[exact("penjualan").to_numpy()] = "SALE"
    movement[exact("return penjualan").to_numpy()] = "SALES_RETURN"
    movement[exact("pembelian pre receive").to_numpy()] = "PRE_RECEIVE"
    movement[exact("pembelian").to_numpy()] = "PURCHASE"

    opname = lower.str.contains("penyesuaian opname|mutasi opname", regex=True)
    movement[(opname & is_in).to_numpy()] = "OPNAME_IN"
    movement[(opname & is_out).to_numpy()] = "OPNAME_OUT"
    movement[(opname & ~(is_in | is_out)).to_numpy()] = "OPNAME_MATCH"

    adjustment = exact("adjustment")
    movement[(adjustment & is_in).to_numpy()] = "ADJUSTMENT_IN"
    movement[(adjustment & is_out).to_numpy()] = "ADJUSTMENT_OUT"
    movement[(adjustment & ~(is_in | is_out)).to_numpy()] = "ADJUSTMENT_MATCH"

    transfer = lower.str.contains("mutasi|transit online", regex=True) & ~opname
    movement[(transfer & is_in).to_numpy()] = "TRANSFER_IN"
    movement[(transfer & is_out).to_numpy()] = "TRANSFER_OUT"
    movement[(transfer & ~(is_in | is_out)).to_numpy()] = "OTHER_MATCH"

    df["movement"] = movement

    # Vectorized extraction where possible.
    partner = text.str.extract(r"(?i)mutasi\s+(?:ke|dari)\s+([^\(]+)", expand=False).fillna("").str.strip()
    df["movement_partner"] = partner
    df["movement_reference"] = text.str.extract(r"\(([^\)]+)\)", expand=False).fillna("").str.strip()

    df["date"] = df["tgl"].dt.normalize()
    df["month"] = df["tgl"].dt.to_period("M").dt.to_timestamp()
    df["week"] = df["date"] - pd.to_timedelta(df["tgl"].dt.weekday, unit="D")
    df["hour"] = df["tgl"].dt.hour
    df["day_name"] = df["tgl"].dt.day_name()

    df["sales_sign"] = np.select([df["movement"].eq("SALE"), df["movement"].eq("SALES_RETURN")], [1.0, -1.0], default=0.0)
    df["net_sales_value"] = df["sales_sign"] * df["subtotal"]
    df["net_sales_qty"] = np.where(df["movement"].eq("SALE"), df["stock_out"], np.where(df["movement"].eq("SALES_RETURN"), -df["stock_in"], 0.0))
    return df


def build_master(opening: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Build one product master while preserving metadata from either source.

    If one source has no sub_kel, the loader supplies a placeholder. We treat that
    placeholder as missing during the merge so a real sub_kel from the other source
    is never overwritten.
    """
    cols = ["sku", "nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]
    opening_master = opening[cols].drop_duplicates("sku", keep="last").set_index("sku")
    tx_master = tx.sort_values("tgl", na_position="first").drop_duplicates("sku", keep="last")[cols].set_index("sku")

    for frame in (opening_master, tx_master):
        for c in ["nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]:
            frame[c] = frame[c].replace({"": pd.NA, SUB_KEL_FALLBACK: pd.NA})

    master = tx_master.combine_first(opening_master).reset_index()
    for c in ["nama_barang", "supplier", "subdept", "kel_barang"]:
        master[c] = master[c].fillna("")
    master["sub_kel"] = master["sub_kel"].fillna(SUB_KEL_FALLBACK)
    return master


def aggregate_opening(opening: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]
    agg = opening.groupby("sku", as_index=False).agg(
        saldo_awal=("saldo_awal", "sum"),
        opening_value=("subtotal", "sum"),
    )
    costs = opening.loc[opening["hrg_beli"].gt(0), ["sku", "hrg_beli"]].drop_duplicates("sku", keep="last").rename(columns={"hrg_beli": "opening_cost"})
    agg = agg.merge(costs, on="sku", how="left")
    agg["opening_cost"] = agg["opening_cost"].fillna(0.0)
    latest_meta = opening.drop_duplicates("sku", keep="last")[["sku"] + meta_cols]
    return agg.merge(latest_meta, on="sku", how="left")


def daily_movement(tx: pd.DataFrame) -> pd.DataFrame:
    return tx.groupby(["date", "sku", "movement"], as_index=False).agg(stock_in=("stock_in", "sum"), stock_out=("stock_out", "sum"), value=("subtotal", "sum"), trx_count=("kd_trx", "nunique"))
