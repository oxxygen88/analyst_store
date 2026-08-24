from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


COST_BEARING_MOVEMENTS = {
    "PURCHASE", "PRE_RECEIVE", "TRANSFER_IN", "ADJUSTMENT_IN", "OPNAME_IN", "OTHER_IN"
}


def _asof_merge(events: pd.DataFrame, sales: pd.DataFrame, prefix: str, cost_col: str, ref_col: str) -> pd.DataFrame:
    if events is None or events.empty:
        out = sales[["_row_id"]].copy()
        out[f"{prefix}_cost"] = np.nan
        out[f"{prefix}_date"] = pd.NaT
        out[f"{prefix}_ref"] = ""
        return out
    e = events.copy()
    e = e.dropna(subset=["sku", "tgl", cost_col])
    e = e[e[cost_col].gt(0)].sort_values(["sku", "tgl", ref_col]).copy()
    e = e.rename(columns={cost_col: f"{prefix}_cost", "tgl": f"{prefix}_date", ref_col: f"{prefix}_ref"})
    # pandas.merge_asof requires the merge key to be globally monotonic.
    e = e.sort_values([f"{prefix}_date", "sku"]).copy()
    s = sales[["_row_id", "sku", "tgl"]].sort_values(["tgl", "sku"]).copy()
    merged = pd.merge_asof(
        s,
        e[["sku", f"{prefix}_date", f"{prefix}_cost", f"{prefix}_ref"]],
        by="sku",
        left_on="tgl",
        right_on=f"{prefix}_date",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged[["_row_id", f"{prefix}_cost", f"{prefix}_date", f"{prefix}_ref"]]


def build_cost_events(tx: pd.DataFrame) -> pd.DataFrame:
    ev = tx[tx["movement"].isin(COST_BEARING_MOVEMENTS) & tx["stock_in"].gt(0) & tx["harga"].gt(0)].copy()
    if ev.empty:
        return pd.DataFrame(columns=["sku", "tgl", "cost", "ref", "source"])
    ev["cost"] = ev["harga"]
    ev["ref"] = np.where(ev["movement_reference"].ne(""), ev["movement_reference"], ev["kd_trx"])
    ev["source"] = ev["movement"].map({
        "PURCHASE": "TX_PURCHASE",
        "PRE_RECEIVE": "PRE_RECEIVE",
        "TRANSFER_IN": "TRANSFER_IN",
        "ADJUSTMENT_IN": "ADJUSTMENT_IN",
        "OPNAME_IN": "OPNAME_IN",
        "OTHER_IN": "OTHER_IN",
    }).fillna("STOCK_IN")
    return ev[["sku", "tgl", "cost", "ref", "source"]]


def resolve_commercial_hpp(
    tx: pd.DataFrame,
    opening: pd.DataFrame,
    purchases: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Resolve cost for SALE and SALES_RETURN rows.

    Priority:
      1) Last purchase-history price on/before transaction date.
      2) Last cost-bearing stock-in on/before transaction date.
      3) Opening stock cost.
      4) First future known cost (new-SKU fallback).
    """
    df = tx.copy()
    df["hpp_unit"] = np.nan
    df["hpp_source"] = "NOT_APPLICABLE"
    df["hpp_reference"] = ""
    df["hpp_date"] = pd.NaT

    commercial_mask = df["movement"].isin(["SALE", "SALES_RETURN"])
    sales = df.loc[commercial_mask].copy().reset_index().rename(columns={"index": "_orig_index"})
    if sales.empty:
        return df
    sales["_row_id"] = np.arange(len(sales))

    # 1. Purchase history as-of.
    if purchases is not None and not purchases.empty:
        p = purchases.copy()
        p = p.rename(columns={"harga_beli": "cost", "no_faktur_beli": "ref"})
        p["source"] = "LAST_PURCHASE"
        purchase_asof = _asof_merge(p, sales, "purchase", "cost", "ref")
    else:
        purchase_asof = sales[["_row_id"]].copy()
        purchase_asof["purchase_cost"] = np.nan
        purchase_asof["purchase_date"] = pd.NaT
        purchase_asof["purchase_ref"] = ""

    # 2. Cost-bearing stock movement as-of.
    cost_ev = build_cost_events(df)
    movement_asof = _asof_merge(cost_ev, sales, "movement", "cost", "ref")

    r = sales[["_row_id", "_orig_index", "sku", "tgl"]].merge(purchase_asof, on="_row_id", how="left")
    r = r.merge(movement_asof, on="_row_id", how="left")

    # Opening cost.
    open_cost = opening.loc[opening["hrg_beli"].gt(0), ["sku", "hrg_beli"]].drop_duplicates("sku", keep="last").rename(columns={"hrg_beli": "opening_cost"})
    r = r.merge(open_cost, on="sku", how="left")

    # Future first-known cost from purchase + movements.
    future_frames = []
    if purchases is not None and not purchases.empty:
        p2 = purchases[["sku", "tgl", "harga_beli", "no_faktur_beli"]].copy()
        p2.columns = ["sku", "tgl", "cost", "ref"]
        p2["source"] = "FUTURE_PURCHASE"
        future_frames.append(p2)
    if not cost_ev.empty:
        m2 = cost_ev.copy()
        m2["source"] = "FUTURE_" + m2["source"].astype(str)
        future_frames.append(m2)
    if future_frames:
        future = pd.concat(future_frames, ignore_index=True)
        future = future.dropna(subset=["tgl", "cost"])
        future = future[future["cost"].gt(0)].sort_values(["sku", "tgl"])
        future_first = future.drop_duplicates("sku", keep="first").rename(columns={
            "tgl": "future_date", "cost": "future_cost", "ref": "future_ref", "source": "future_source"
        })
        r = r.merge(future_first[["sku", "future_date", "future_cost", "future_ref", "future_source"]], on="sku", how="left")
    else:
        r["future_date"] = pd.NaT
        r["future_cost"] = np.nan
        r["future_ref"] = ""
        r["future_source"] = "UNRESOLVED"

    # Priority resolution.
    r["hpp_unit"] = r["purchase_cost"]
    r["hpp_source"] = np.where(r["purchase_cost"].notna(), "LAST_PURCHASE", "")
    r["hpp_reference"] = np.where(r["purchase_cost"].notna(), r["purchase_ref"].fillna(""), "")
    r["hpp_date"] = r["purchase_date"]

    miss = r["hpp_unit"].isna() & r["movement_cost"].notna()
    r.loc[miss, "hpp_unit"] = r.loc[miss, "movement_cost"]
    r.loc[miss, "hpp_source"] = "STOCK_IN_COST"
    r.loc[miss, "hpp_reference"] = r.loc[miss, "movement_ref"].fillna("")
    r.loc[miss, "hpp_date"] = r.loc[miss, "movement_date"]

    miss = r["hpp_unit"].isna() & r["opening_cost"].notna() & r["opening_cost"].gt(0)
    r.loc[miss, "hpp_unit"] = r.loc[miss, "opening_cost"]
    r.loc[miss, "hpp_source"] = "OPENING_STOCK"
    r.loc[miss, "hpp_reference"] = "OPENING_2026"

    # Future cost only if no historical/opening cost exists.
    miss = r["hpp_unit"].isna() & r["future_cost"].notna()
    r.loc[miss, "hpp_unit"] = r.loc[miss, "future_cost"]
    r.loc[miss, "hpp_source"] = r.loc[miss, "future_source"].fillna("FUTURE_COST")
    r.loc[miss, "hpp_reference"] = r.loc[miss, "future_ref"].fillna("")
    r.loc[miss, "hpp_date"] = r.loc[miss, "future_date"]

    r.loc[r["hpp_unit"].isna(), "hpp_source"] = "UNRESOLVED"

    resolved = r.set_index("_orig_index")[["hpp_unit", "hpp_source", "hpp_reference", "hpp_date"]]
    df.loc[resolved.index, "hpp_unit"] = resolved["hpp_unit"]
    df.loc[resolved.index, "hpp_source"] = resolved["hpp_source"]
    df.loc[resolved.index, "hpp_reference"] = resolved["hpp_reference"]
    df.loc[resolved.index, "hpp_date"] = resolved["hpp_date"]

    # Signed HPP aligns with net sales sign; returns reduce COGS.
    df["net_hpp"] = np.where(
        df["movement"].eq("SALE"), df["stock_out"] * df["hpp_unit"],
        np.where(df["movement"].eq("SALES_RETURN"), -df["stock_in"] * df["hpp_unit"], 0.0)
    )
    df["gross_profit"] = df["net_sales_value"] - df["net_hpp"].fillna(0.0)
    return df


def latest_cost_asof(
    as_of_date: pd.Timestamp,
    master_skus: pd.Series,
    tx: pd.DataFrame,
    opening: pd.DataFrame,
    purchases: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Latest estimated unit cost per SKU as of a selected date."""
    as_of = pd.Timestamp(as_of_date)
    frames = []
    if purchases is not None and not purchases.empty:
        p = purchases[purchases["tgl"].le(as_of) & purchases["harga_beli"].gt(0)][["sku", "tgl", "harga_beli", "no_faktur_beli"]].copy()
        p.columns = ["sku", "tgl", "cost", "ref"]
        p["source"] = "LAST_PURCHASE"
        frames.append(p)
    m = build_cost_events(tx)
    if not m.empty:
        m = m[m["tgl"].le(as_of)].copy()
        frames.append(m)
    if frames:
        ev = pd.concat(frames, ignore_index=True).sort_values(["sku", "tgl"])
        last = ev.drop_duplicates("sku", keep="last")[["sku", "cost", "source", "ref", "tgl"]]
    else:
        last = pd.DataFrame(columns=["sku", "cost", "source", "ref", "tgl"])
    base = pd.DataFrame({"sku": pd.Series(master_skus).astype(str).unique()})
    out = base.merge(last, on="sku", how="left")
    opening_cost = opening.loc[opening["hrg_beli"].gt(0), ["sku", "hrg_beli"]].drop_duplicates("sku", keep="last").rename(columns={"hrg_beli": "opening_cost"})
    out = out.merge(opening_cost, on="sku", how="left")
    miss = out["cost"].isna() & out["opening_cost"].notna() & out["opening_cost"].gt(0)
    out.loc[miss, "cost"] = out.loc[miss, "opening_cost"]
    out.loc[miss, "source"] = "OPENING_STOCK"
    out.loc[miss, "ref"] = "OPENING_2026"
    return out[["sku", "cost", "source", "ref", "tgl"]]
