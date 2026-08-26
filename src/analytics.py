from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from .config import (
    HIGH_COVER_DAYS,
    HEALTHY_COVER_DAYS,
    LOW_COVER_DAYS,
    OPPORTUNITY_WEIGHTS,
    PARETO_CORE_SHARE,
    PARETO_REVENUE_SHARE,
)
from .hpp import latest_cost_asof
from .transform import build_master
from .utils import safe_div


@dataclass
class TargetStatus:
    month: pd.Timestamp
    target: float
    actual: float
    achievement: float
    gap: float
    elapsed_days: int
    remaining_days: int
    daily_run_rate: float
    required_daily_sales: float
    projected_month_end: float
    projected_gap: float
    pace_achievement: float


def filter_period(tx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return tx[tx["date"].between(s.normalize(), e.normalize())].copy()


def commercial_kpis(tx_period: pd.DataFrame) -> Dict[str, float]:
    sale = tx_period[tx_period["movement"].eq("SALE")]
    ret = tx_period[tx_period["movement"].eq("SALES_RETURN")]
    gross_sales = float(sale["subtotal"].sum())
    return_value = float(ret["subtotal"].sum())
    net_sales = gross_sales - return_value
    sale_qty = float(sale["stock_out"].sum())
    return_qty = float(ret["stock_in"].sum())
    net_qty = sale_qty - return_qty
    trx_count = int(sale["kd_trx"].nunique())
    atv = safe_div(net_sales, trx_count, 0.0)
    upt = safe_div(net_qty, trx_count, 0.0)
    net_hpp = float(tx_period.loc[tx_period["movement"].isin(["SALE", "SALES_RETURN"]), "net_hpp"].sum(min_count=1)) if "net_hpp" in tx_period.columns else np.nan
    if np.isnan(net_hpp):
        gp = np.nan
        margin = np.nan
    else:
        gp = net_sales - net_hpp
        margin = safe_div(gp, net_sales, np.nan)
    return {
        "gross_sales": gross_sales,
        "return_value": return_value,
        "net_sales": net_sales,
        "sale_qty": sale_qty,
        "return_qty": return_qty,
        "net_qty": net_qty,
        "trx_count": trx_count,
        "atv": atv,
        "upt": upt,
        "net_hpp": net_hpp,
        "gross_profit": gp,
        "gross_margin": margin,
    }


def monthly_sales(tx: pd.DataFrame) -> pd.DataFrame:
    commercial = tx[tx["movement"].isin(["SALE", "SALES_RETURN"])].copy()
    out = commercial.groupby("month", as_index=False).agg(
        net_sales=("net_sales_value", "sum"),
        net_qty=("net_sales_qty", "sum"),
        net_hpp=("net_hpp", "sum") if "net_hpp" in commercial.columns else ("net_sales_qty", lambda s: np.nan),
    )
    sale_trx = tx[tx["movement"].eq("SALE")].groupby("month")["kd_trx"].nunique().rename("trx_count")
    out = out.merge(sale_trx, on="month", how="left")
    out["trx_count"] = out["trx_count"].fillna(0).astype(int)
    out["atv"] = out["net_sales"] / out["trx_count"].replace(0, np.nan)
    out["upt"] = out["net_qty"] / out["trx_count"].replace(0, np.nan)
    if "net_hpp" in out.columns:
        out["gross_profit"] = out["net_sales"] - out["net_hpp"]
        out["gross_margin"] = out["gross_profit"] / out["net_sales"].replace(0, np.nan)
    return out.sort_values("month")


def target_status(tx: pd.DataFrame, targets: Optional[pd.DataFrame], month: pd.Timestamp, as_of_date: pd.Timestamp, location: str = "IDK-ATP") -> Optional[TargetStatus]:
    if targets is None or targets.empty:
        return None
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    as_of = min(pd.Timestamp(as_of_date).normalize(), month.to_period("M").to_timestamp("M"))
    row = targets[(targets["bulan"].eq(month)) & (targets["lokasi"].astype(str).eq(str(location)))]
    if row.empty:
        row = targets[targets["bulan"].eq(month)]
    if row.empty:
        return None
    target = float(row.iloc[0]["target_omzet"])
    start = month
    actual = float(tx[tx["date"].between(start, as_of)]["net_sales_value"].sum())
    days_in_month = calendar.monthrange(month.year, month.month)[1]
    elapsed = max(1, min(as_of.day, days_in_month))
    remaining = max(days_in_month - elapsed, 0)
    run_rate = actual / elapsed
    gap = max(target - actual, 0.0)
    required = gap / remaining if remaining > 0 else gap
    projected = actual if remaining == 0 else run_rate * days_in_month
    projected_gap = max(target - projected, 0.0)
    expected_to_date = target * elapsed / days_in_month
    pace = safe_div(actual, expected_to_date, np.nan)
    return TargetStatus(
        month=month, target=target, actual=actual, achievement=safe_div(actual, target, np.nan), gap=gap,
        elapsed_days=elapsed, remaining_days=remaining, daily_run_rate=run_rate,
        required_daily_sales=required, projected_month_end=projected, projected_gap=projected_gap,
        pace_achievement=pace,
    )


def current_stock_snapshot(opening: pd.DataFrame, tx: pd.DataFrame, as_of_date: pd.Timestamp, purchases: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of_date).normalize()
    meta_cols = ["nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]
    opening_agg = opening.groupby("sku", as_index=False).agg(
        opening_qty=("saldo_awal", "sum"),
        opening_value=("subtotal", "sum"),
    )
    tx_upto = tx[tx["date"].le(as_of)]
    movement = tx_upto.groupby("sku", as_index=False).agg(
        stock_in=("stock_in", "sum"),
        stock_out=("stock_out", "sum"),
    )
    master = build_master(opening, tx_upto)
    snap = master.merge(opening_agg, on="sku", how="left").merge(movement, on="sku", how="left")
    for c in ["opening_qty", "opening_value", "stock_in", "stock_out"]:
        snap[c] = snap[c].fillna(0.0)
    snap["current_stock"] = snap["opening_qty"] + snap["stock_in"] - snap["stock_out"]
    costs = latest_cost_asof(as_of, snap["sku"], tx, opening, purchases)
    snap = snap.merge(costs.rename(columns={"cost":"current_cost", "source":"cost_source", "ref":"cost_reference", "tgl":"cost_date"}), on="sku", how="left")
    snap["current_stock_value"] = snap["current_stock"] * snap["current_cost"].fillna(0.0)
    return snap


def _active_start_by_sku(opening: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    opening_skus = set(opening["sku"].astype(str))
    first_in = tx[tx["stock_in"].gt(0)].groupby("sku", as_index=False)["date"].min().rename(columns={"date":"first_in_date"})
    all_skus = pd.DataFrame({"sku": pd.Index(pd.concat([opening["sku"], tx["sku"]]).astype(str).unique())})
    all_skus = all_skus.merge(first_in, on="sku", how="left")
    first_year = int(tx["date"].dropna().min().year) if tx["date"].notna().any() else 2026
    jan1 = pd.Timestamp(year=first_year, month=1, day=1)
    all_skus["active_start"] = np.where(all_skus["sku"].isin(opening_skus), jan1, all_skus["first_in_date"])
    all_skus["active_start"] = pd.to_datetime(all_skus["active_start"], errors="coerce")
    return all_skus[["sku", "active_start"]]


def inventory_health(opening: pd.DataFrame, tx: pd.DataFrame, as_of_date: pd.Timestamp, purchases: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of_date).normalize()
    snap = current_stock_snapshot(opening, tx, as_of, purchases)
    active = _active_start_by_sku(opening, tx)
    snap = snap.merge(active, on="sku", how="left")

    # Last complete month is used for sigma baseline to avoid partial-month distortion.
    current_month_start = as_of.to_period("M").to_timestamp()
    first_month = pd.Timestamp(year=as_of.year, month=1, day=1)
    last_complete_month = (current_month_start - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    complete_months = pd.date_range(first_month, last_complete_month, freq="MS") if last_complete_month >= first_month else pd.DatetimeIndex([])

    # Vectorized monthly baseline. New SKUs only enter the baseline from their activation month.
    active_stats = active.copy()
    active_stats["active_month"] = pd.to_datetime(active_stats["active_start"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    active_stats["active_month"] = active_stats["active_month"].fillna(first_month)

    if len(complete_months):
        months_df = pd.DataFrame({"month": complete_months})
        grid = active_stats[["sku","active_month"]].merge(months_df, how="cross")
        grid = grid[grid["month"].ge(grid["active_month"])]
        sales_complete = tx[
            tx["movement"].isin(["SALE","SALES_RETURN"]) &
            tx["month"].isin(complete_months)
        ].groupby(["sku","month"], as_index=False)["net_sales_qty"].sum()
        grid = grid.merge(sales_complete, on=["sku","month"], how="left")
        grid["net_sales_qty"] = grid["net_sales_qty"].fillna(0.0)
        grid["qty_sq"] = grid["net_sales_qty"] ** 2
        stats = grid.groupby("sku", as_index=False).agg(
            qty_sum=("net_sales_qty","sum"),
            qty_sq_sum=("qty_sq","sum"),
            baseline_months=("month","count"),
        )
        stats["avg_monthly_sales"] = stats["qty_sum"] / stats["baseline_months"].replace(0,np.nan)
        variance = stats["qty_sq_sum"] / stats["baseline_months"].replace(0,np.nan) - stats["avg_monthly_sales"] ** 2
        stats["std_monthly_sales"] = np.sqrt(variance.clip(lower=0))
        stats["baseline_sales_qty"] = stats["qty_sum"]
        stats = stats[["sku","avg_monthly_sales","std_monthly_sales","baseline_sales_qty","baseline_months"]]
    else:
        stats = active_stats[["sku"]].copy()
        stats["avg_monthly_sales"] = 0.0
        stats["std_monthly_sales"] = 0.0
        stats["baseline_sales_qty"] = 0.0
        stats["baseline_months"] = 0

    snap = snap.merge(stats, on="sku", how="left")
    for c in ["avg_monthly_sales","std_monthly_sales","baseline_sales_qty","baseline_months"]:
        snap[c] = snap[c].fillna(0)

    # Recency and current demand.
    sales_positive = tx[(tx["movement"].eq("SALE")) & tx["date"].le(as_of)]
    last_sale = sales_positive.groupby("sku", as_index=False)["date"].max().rename(columns={"date":"last_sale_date"})
    sales30 = tx[(tx["movement"].isin(["SALE", "SALES_RETURN"])) & tx["date"].between(as_of - pd.Timedelta(days=29), as_of)]
    sales30 = sales30.groupby("sku", as_index=False)["net_sales_qty"].sum().rename(columns={"net_sales_qty":"sales_qty_30d"})
    snap = snap.merge(last_sale, on="sku", how="left").merge(sales30, on="sku", how="left")
    snap["sales_qty_30d"] = snap["sales_qty_30d"].fillna(0.0)
    snap["avg_daily_sales_30d"] = snap["sales_qty_30d"].clip(lower=0) / 30.0
    snap["stock_cover_days"] = snap["current_stock"] / snap["avg_daily_sales_30d"].replace(0, np.nan)
    snap["days_since_last_sale"] = (as_of - pd.to_datetime(snap["last_sale_date"])).dt.days

    snap["slow_threshold"] = snap["avg_monthly_sales"] + snap["std_monthly_sales"]
    snap["dead_threshold"] = snap["avg_monthly_sales"] + 3 * snap["std_monthly_sales"]
    snap["overstock_threshold"] = snap["avg_monthly_sales"] + 6 * snap["std_monthly_sales"]

    # Mutually-exclusive status bands.
    st = snap["current_stock"]
    no_sales = snap["baseline_sales_qty"].le(0) & snap["sales_qty_30d"].le(0)
    conditions = [
        st.lt(0),
        st.eq(0) & snap["sales_qty_30d"].gt(0),
        st.eq(0),
        st.gt(0) & no_sales,
        st.gt(0) & snap["overstock_threshold"].gt(0) & st.ge(snap["overstock_threshold"]),
        st.gt(0) & snap["dead_threshold"].gt(0) & st.ge(snap["dead_threshold"]),
        st.gt(0) & snap["slow_threshold"].gt(0) & st.ge(snap["slow_threshold"]),
    ]
    choices = ["NEGATIVE","STOCKOUT","ZERO_STOCK","NO_SALES","OVERSTOCK","DEAD","SLOW"]
    snap["inventory_status"] = np.select(conditions, choices, default="NORMAL")
    return snap

def pareto_products(tx: pd.DataFrame, inventory: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    p = filter_period(tx, start, end)
    commercial = p[p["movement"].isin(["SALE", "SALES_RETURN"])].copy()
    agg_map = {
        "revenue": ("net_sales_value", "sum"),
        "qty": ("net_sales_qty", "sum"),
    }
    if "net_hpp" in commercial.columns:
        agg_map["hpp"] = ("net_hpp", "sum")
    prod = commercial.groupby("sku", as_index=False).agg(**agg_map)
    prod = prod[prod["revenue"].gt(0)].sort_values("revenue", ascending=False).reset_index(drop=True)
    if prod.empty:
        return prod
    prod["rank"] = np.arange(1, len(prod)+1)
    prod["revenue_share"] = prod["revenue"] / prod["revenue"].sum()
    prod["cumulative_share"] = prod["revenue_share"].cumsum()
    core_n = max(1, int(np.ceil(len(prod) * PARETO_CORE_SHARE)))
    a80_n = int(np.searchsorted(prod["cumulative_share"].values, PARETO_REVENUE_SHARE, side="left") + 1)
    prod["pareto_group"] = np.select(
        [prod["rank"].le(core_n), prod["rank"].le(a80_n)],
        ["CORE_20", "OPPORTUNITY"],
        default="LONG_TAIL",
    )
    prod["a80_member"] = prod["rank"].le(a80_n)
    if "hpp" in prod.columns:
        prod["gross_profit"] = prod["revenue"] - prod["hpp"]
        prod["gross_margin"] = prod["gross_profit"] / prod["revenue"].replace(0, np.nan)
    meta_cols = ["sku", "nama_barang", "supplier", "subdept", "kel_barang", "sub_kel", "current_stock", "current_stock_value", "stock_cover_days", "inventory_status", "current_cost"]
    have = [c for c in meta_cols if c in inventory.columns]
    prod = prod.merge(inventory[have], on="sku", how="left")
    return prod


def _recent_growth_by_sku(tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of).normalize()
    cur_start = as_of - pd.Timedelta(days=29)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=29)
    c = tx[tx["date"].between(cur_start, as_of)].groupby("sku")["net_sales_value"].sum().rename("rev_30d")
    p = tx[tx["date"].between(prev_start, prev_end)].groupby("sku")["net_sales_value"].sum().rename("rev_prev_30d")
    out = pd.concat([c,p], axis=1).fillna(0).reset_index()
    out["growth_30d"] = (out["rev_30d"] - out["rev_prev_30d"]) / out["rev_prev_30d"].replace(0, np.nan)
    out.loc[(out["rev_prev_30d"].eq(0)) & (out["rev_30d"].gt(0)), "growth_30d"] = 1.0
    out["growth_30d"] = out["growth_30d"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1, 2)
    return out


def _consistency_by_sku(tx: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    end = pd.Timestamp(end)
    start = (end.to_period("M").to_timestamp() - pd.DateOffset(months=5)).normalize()
    x = tx[(tx["date"].between(start, end)) & tx["movement"].isin(["SALE","SALES_RETURN"])].groupby(["sku","month"], as_index=False)["net_sales_value"].sum()
    if x.empty:
        return pd.DataFrame(columns=["sku","consistency_score"])
    s = x.groupby("sku")["net_sales_value"].agg(["mean","std"]).reset_index()
    s["cv"] = s["std"].fillna(0) / s["mean"].abs().replace(0, np.nan)
    s["consistency_score"] = (100 / (1 + s["cv"].fillna(2))).clip(0,100)
    return s[["sku","consistency_score"]]


def opportunity_scoring(pareto: pd.DataFrame, tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if pareto.empty:
        return pareto
    df = pareto.copy()
    growth = _recent_growth_by_sku(tx, as_of)
    consistency = _consistency_by_sku(tx, as_of)
    df = df.merge(growth, on="sku", how="left").merge(consistency, on="sku", how="left")
    df["growth_30d"] = df["growth_30d"].fillna(0)
    df["consistency_score"] = df["consistency_score"].fillna(50)
    df["revenue_score"] = df["revenue"].rank(pct=True) * 100
    df["growth_score"] = ((df["growth_30d"] + 1) / 3 * 100).clip(0,100)

    if "gross_margin" in df.columns and df["gross_margin"].notna().any():
        df["margin_score"] = df["gross_margin"].rank(pct=True) * 100
        weights = OPPORTUNITY_WEIGHTS.copy()
    else:
        df["margin_score"] = np.nan
        weights = OPPORTUNITY_WEIGHTS.copy()
        redistribute = weights.pop("margin")
        base = sum(weights.values())
        weights = {k: v + redistribute * (v/base) for k,v in weights.items()}
        weights["margin"] = 0.0

    cover = df["stock_cover_days"] if "stock_cover_days" in df.columns else pd.Series(np.nan, index=df.index)
    stock = df["current_stock"] if "current_stock" in df.columns else pd.Series(0, index=df.index)
    stock_score = np.select(
        [stock.le(0), cover.lt(LOW_COVER_DAYS), cover.between(LOW_COVER_DAYS, HEALTHY_COVER_DAYS, inclusive="both"), cover.between(HEALTHY_COVER_DAYS, HIGH_COVER_DAYS, inclusive="right"), cover.gt(HIGH_COVER_DAYS), cover.isna() & stock.gt(0)],
        [0, 35, 100, 80, 55, 60], default=50
    )
    df["stock_readiness_score"] = stock_score.astype(float)

    df["opportunity_score"] = (
        weights["revenue"] * df["revenue_score"] +
        weights["growth"] * df["growth_score"] +
        weights["margin"] * df["margin_score"].fillna(0) +
        weights["stock"] * df["stock_readiness_score"] +
        weights["consistency"] * df["consistency_score"]
    ).clip(0,100)

    def action(r):
        g = r.pareto_group
        st = float(r.get("current_stock", 0) or 0)
        status = str(r.get("inventory_status", ""))
        cover_v = r.get("stock_cover_days", np.nan)
        if g == "CORE_20" and st <= 0:
            return "Emergency replenish"
        if g == "CORE_20" and pd.notna(cover_v) and cover_v < LOW_COVER_DAYS:
            return "Protect sales / replenish"
        if g == "CORE_20":
            return "Maintain availability"
        if g == "OPPORTUNITY" and status in {"OVERSTOCK","DEAD","SLOW"} and st > 0:
            return "Push / campaign candidate"
        if g == "OPPORTUNITY" and r.opportunity_score >= 70 and st > 0:
            return "Push sales"
        if g == "OPPORTUNITY" and st <= 0:
            return "Replenish before push"
        if g == "LONG_TAIL" and status in {"OVERSTOCK","DEAD"}:
            return "Reduce buy / transfer / clearance"
        if g == "LONG_TAIL" and status == "SLOW":
            return "Monitor / reduce buy"
        return "Maintain"

    df["recommended_action"] = df.apply(action, axis=1)
    return df.sort_values(["pareto_group","opportunity_score"], ascending=[True,False])


def revenue_inventory_matrix(tx: pd.DataFrame, inventory: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, dimension: str) -> pd.DataFrame:
    p = filter_period(tx, start, end)
    commercial = p[p["movement"].isin(["SALE","SALES_RETURN"])]
    revenue = commercial.groupby(dimension, dropna=False, as_index=False)["net_sales_value"].sum().rename(columns={"net_sales_value":"revenue"})
    inv = inventory.groupby(dimension, dropna=False, as_index=False)["current_stock_value"].sum().rename(columns={"current_stock_value":"inventory_value"})
    out = revenue.merge(inv, on=dimension, how="outer").fillna(0)
    out["revenue_share"] = out["revenue"] / out["revenue"].sum() if out["revenue"].sum() else 0
    positive_inv = out["inventory_value"].clip(lower=0)
    out["inventory_share"] = positive_inv / positive_inv.sum() if positive_inv.sum() else 0
    out["productivity_index"] = out["revenue_share"] / out["inventory_share"].replace(0, np.nan)
    return out.sort_values("revenue", ascending=False)


def anomaly_tables(opening: pd.DataFrame, tx: pd.DataFrame, inventory: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    issues = {}
    issues["negative_opening"] = opening[opening["saldo_awal"].lt(0)].copy()
    issues["negative_current"] = inventory[inventory["current_stock"].lt(0)].copy()
    qty = tx["stock_in"].abs() + tx["stock_out"].abs()
    issues["zero_price"] = tx[(qty.gt(0)) & tx["harga"].le(0)].copy()
    expected = np.where(tx["stock_in"].gt(0), tx["stock_in"] * tx["harga"], tx["stock_out"] * tx["harga"])
    issues["subtotal_mismatch"] = tx[(qty.gt(0)) & (~np.isclose(tx["subtotal"], expected, rtol=1e-6, atol=1.0))].copy()
    issues["both_in_out"] = tx[tx["stock_in"].gt(0) & tx["stock_out"].gt(0)].copy()
    issues["unknown_movement"] = tx[tx["movement"].isin(["OTHER_IN","OTHER_OUT","OTHER_MATCH"])].copy()
    if "date_parse_status" in tx.columns:
        issues["date_unresolved"] = tx[tx["date_parse_status"].isin(["UNRESOLVED", "OUTSIDE_ANALYSIS_YEAR"])].copy()
    else:
        issues["date_unresolved"] = tx[tx["tgl"].isna()].copy()
    dup_cols = [c for c in ["kd_trx","tgl","sku","stock_in","stock_out","harga","subtotal","keterangan"] if c in tx.columns]
    issues["suspicious_duplicates"] = tx[tx.duplicated(dup_cols, keep=False)].copy()
    if "hpp_source" in tx.columns:
        issues["missing_hpp"] = tx[tx["movement"].isin(["SALE","SALES_RETURN"]) & tx["hpp_source"].eq("UNRESOLVED")].copy()
    else:
        issues["missing_hpp"] = pd.DataFrame()
    return issues


def hpp_coverage(tx: pd.DataFrame) -> pd.DataFrame:
    if "hpp_source" not in tx.columns:
        return pd.DataFrame(columns=["hpp_source","lines","revenue","share_lines"])
    x = tx[tx["movement"].isin(["SALE","SALES_RETURN"])].copy()
    if x.empty:
        return pd.DataFrame(columns=["hpp_source","lines","revenue","share_lines"])
    out = x.groupby("hpp_source", as_index=False).agg(lines=("sku","size"), revenue=("net_sales_value", lambda s: float(np.abs(s).sum())))
    out["share_lines"] = out["lines"] / out["lines"].sum()
    return out.sort_values("lines", ascending=False)
