from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .analytics import (
    anomaly_tables,
    commercial_kpis,
    filter_period,
    inventory_health,
    monthly_sales,
    opportunity_scoring,
    pareto_products,
    revenue_inventory_matrix,
    target_status,
)
from .ai_presentation import _module_available, _responses_output_text


MONTH_ALIASES = {
    "januari": 1, "january": 1, "jan": 1,
    "februari": 2, "february": 2, "feb": 2,
    "maret": 3, "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5, "may": 5,
    "juni": 6, "june": 6, "jun": 6,
    "juli": 7, "july": 7, "jul": 7,
    "agustus": 8, "august": 8, "agu": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "october": 10, "okt": 10, "oct": 10,
    "november": 11, "nov": 11,
    "desember": 12, "december": 12, "des": 12, "dec": 12,
}

STATUS_ALIASES = {
    "overstock": "OVERSTOCK",
    "dead stock": "DEAD",
    "dead": "DEAD",
    "slow moving": "SLOW",
    "slow": "SLOW",
    "stockout": "STOCKOUT",
    "stock out": "STOCKOUT",
    "stok kosong": "STOCKOUT",
    "negative": "NEGATIVE",
    "negatif": "NEGATIVE",
    "no sales": "NO_SALES",
    "tidak pernah terjual": "NO_SALES",
    "zero stock": "ZERO_STOCK",
    "stok nol": "ZERO_STOCK",
    "normal": "NORMAL",
}

PARETO_ALIASES = {
    "core 20": "CORE_20",
    "core20": "CORE_20",
    "core": "CORE_20",
    "opportunity": "OPPORTUNITY",
    "potensial": "OPPORTUNITY",
    "long tail": "LONG_TAIL",
}


def _jsonable(value: Any):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _records(df: Optional[pd.DataFrame], limit: int = 100) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.head(limit).copy()
    rows = []
    for row in out.to_dict("records"):
        rows.append({k: _jsonable(v) for k, v in row.items()})
    return rows


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def _detected_values(question: str, values: Iterable[Any], limit: int = 8) -> list[str]:
    q = _normalize_text(question)
    candidates = []
    for raw in pd.Series(list(values), dtype="object").dropna().astype(str).drop_duplicates().tolist():
        val = raw.strip()
        if len(val) < 2:
            continue
        norm_val = _normalize_text(val)
        if len(norm_val) <= 3:
            if re.search(rf"(?<!\w){re.escape(norm_val)}(?!\w)", q):
                candidates.append(val)
        elif norm_val in q:
            candidates.append(val)
    candidates.sort(key=len, reverse=True)
    return candidates[:limit]


def detect_entities(question: str, master: pd.DataFrame) -> Dict[str, list[str]]:
    """Detect exact business entities mentioned in natural-language questions."""
    text = str(question or "")
    out: Dict[str, list[str]] = {}
    if master is None or master.empty:
        return out

    # SKU matching is exact-token based so leading zeros are preserved.
    sku_values = set(master["sku"].dropna().astype(str)) if "sku" in master.columns else set()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]{2,}", text)
    skus = [tok for tok in tokens if tok in sku_values]
    if skus:
        out["sku"] = list(dict.fromkeys(skus))[:10]

    for col in ["supplier", "subdept", "kel_barang", "sub_kel", "nama_barang"]:
        if col in master.columns:
            found = _detected_values(text, master[col].dropna().unique(), limit=8 if col != "nama_barang" else 5)
            if found:
                out[col] = found
    return out


def _period_from_month(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end


def resolve_question_period(
    question: str,
    as_of: pd.Timestamp,
    min_date: pd.Timestamp,
    previous_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a user question into a deterministic analysis period."""
    q = _normalize_text(question)
    as_of = pd.Timestamp(as_of).normalize()
    min_date = pd.Timestamp(min_date).normalize()
    requested_start: Optional[pd.Timestamp] = None
    requested_end: Optional[pd.Timestamp] = None
    label = "MTD"

    # Explicit ISO dates take precedence.
    iso_dates = [pd.Timestamp(x) for x in re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", q)]
    if len(iso_dates) >= 2:
        requested_start, requested_end = min(iso_dates[0], iso_dates[1]), max(iso_dates[0], iso_dates[1])
        label = "custom date range"
    elif len(iso_dates) == 1:
        requested_start = requested_end = iso_dates[0]
        label = "specific date"
    else:
        # Relative rolling windows.
        m = re.search(r"\b(7|14|30|60|90|180)\s*(?:hari|days?)\s*(?:terakhir|last)?\b", q)
        if m:
            days = int(m.group(1))
            requested_end = as_of
            requested_start = as_of - pd.Timedelta(days=days - 1)
            label = f"last {days} days"
        elif any(x in q for x in ["ytd", "year to date", "tahun berjalan", "tahun ini"]):
            requested_start = pd.Timestamp(year=as_of.year, month=1, day=1)
            requested_end = as_of
            label = "YTD"
        elif "minggu ini" in q or "this week" in q:
            requested_start = as_of - pd.Timedelta(days=as_of.weekday())
            requested_end = as_of
            label = "this week"
        else:
            # Month names, including ranges such as Januari sampai Maret.
            month_hits = []
            for name, month in MONTH_ALIASES.items():
                for mt in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", q):
                    month_hits.append((mt.start(), month, name))
            month_hits.sort()
            years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
            year = years[0] if years else as_of.year
            if month_hits:
                first_month = month_hits[0][1]
                last_month = month_hits[-1][1]
                requested_start = pd.Timestamp(year=year, month=first_month, day=1)
                requested_end = pd.Timestamp(year=year, month=last_month, day=1) + pd.offsets.MonthEnd(0)
                label = f"{requested_start:%b %Y}" if first_month == last_month else f"{requested_start:%b}–{requested_end:%b %Y}"

    if requested_start is None or requested_end is None:
        # Follow-up questions can inherit the last scope if no new period was mentioned.
        followup_markers = ["itu", "tersebut", "tadi", "yang sama", "dari sana", "lalu", "kalau", "bagaimana dengan"]
        if previous_scope and any(x in q for x in followup_markers):
            try:
                requested_start = pd.Timestamp(previous_scope["period"]["start"])
                requested_end = pd.Timestamp(previous_scope["period"]["end"])
                label = "follow-up previous period"
            except Exception:
                requested_start = None
        if requested_start is None or requested_end is None:
            requested_start = as_of.to_period("M").to_timestamp()
            requested_end = as_of
            label = "MTD"

    requested_start = pd.Timestamp(requested_start).normalize()
    requested_end = pd.Timestamp(requested_end).normalize()
    effective_start = max(requested_start, min_date)
    effective_end = min(requested_end, as_of)
    available = effective_start <= effective_end

    return {
        "label": label,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "start": effective_start,
        "end": effective_end,
        "data_available": bool(available),
        "clipped_to_data": bool(requested_start < min_date or requested_end > as_of),
    }


def _merge_filters(global_filters: Optional[Dict[str, list[str]]], detected: Dict[str, list[str]]) -> Dict[str, list[str]]:
    merged: Dict[str, list[str]] = {}
    for source in [global_filters or {}, detected or {}]:
        for col, vals in source.items():
            vals = [str(v) for v in vals if str(v).strip()]
            if not vals:
                continue
            if col in merged:
                # If both sidebar and question specify a dimension, use the intersection when possible.
                inter = [v for v in merged[col] if v in set(vals)]
                merged[col] = inter if inter else vals
            else:
                merged[col] = vals
    return merged


def _apply_filters(df: pd.DataFrame, filters: Dict[str, list[str]]) -> pd.DataFrame:
    out = df
    for col, vals in (filters or {}).items():
        if vals and col in out.columns:
            out = out[out[col].astype(str).isin([str(v) for v in vals])]
    return out.copy()


def _sales_group(tx_period: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    commercial = tx_period[tx_period["movement"].isin(["SALE", "SALES_RETURN"])].copy()
    if commercial.empty:
        return pd.DataFrame(columns=group_cols + ["net_sales", "net_qty", "trx_count", "gross_profit", "gross_margin"])

    group_cols = [c for c in group_cols if c in commercial.columns]
    if not group_cols:
        return pd.DataFrame([commercial_kpis(commercial)])

    grouped = commercial.groupby(group_cols, dropna=False, as_index=False).agg(
        net_sales=("net_sales_value", "sum"),
        net_qty=("net_sales_qty", "sum"),
    )
    sale_trx = commercial[commercial["movement"].eq("SALE")].groupby(group_cols, dropna=False)["kd_trx"].nunique().rename("trx_count").reset_index()
    grouped = grouped.merge(sale_trx, on=group_cols, how="left")
    grouped["trx_count"] = grouped["trx_count"].fillna(0).astype(int)
    grouped["atv"] = grouped["net_sales"] / grouped["trx_count"].replace(0, np.nan)
    grouped["upt"] = grouped["net_qty"] / grouped["trx_count"].replace(0, np.nan)
    if "net_hpp" in commercial.columns:
        hpp = commercial.groupby(group_cols, dropna=False)["net_hpp"].sum(min_count=1).rename("net_hpp").reset_index()
        grouped = grouped.merge(hpp, on=group_cols, how="left")
        grouped["gross_profit"] = grouped["net_sales"] - grouped["net_hpp"]
        grouped["gross_margin"] = grouped["gross_profit"] / grouped["net_sales"].replace(0, np.nan)
    return grouped.sort_values("net_sales", ascending=False)


def _movement_summary(tx_period: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if tx_period.empty:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in tx_period.columns]
    if not group_cols:
        group_cols = ["movement"]
    out = tx_period.groupby(group_cols, dropna=False, as_index=False).agg(
        stock_in=("stock_in", "sum"),
        stock_out=("stock_out", "sum"),
        movement_value=("subtotal", "sum"),
        trx_count=("kd_trx", "nunique"),
    )
    return out.sort_values("movement_value", ascending=False)


def _status_filters_from_question(question: str) -> list[str]:
    q = _normalize_text(question)
    out = []
    for alias, status in STATUS_ALIASES.items():
        if alias in q and status not in out:
            out.append(status)
    return out


def _pareto_filters_from_question(question: str) -> list[str]:
    q = _normalize_text(question)
    out = []
    for alias, status in PARETO_ALIASES.items():
        if alias in q and status not in out:
            out.append(status)
    return out


def _domain_flags(question: str) -> Dict[str, bool]:
    q = _normalize_text(question)
    def has(*words):
        return any(w in q for w in words)
    return {
        "sales": has("sales", "revenue", "omzet", "penjualan", "transaksi", "atv", "upt", "qty", "unit"),
        "target": has("target", "achievement", "gap", "pace", "proyeksi", "projection", "capai", "mengejar"),
        "product": has("produk", "product", "item", "sku", "barang", "pareto", "core", "opportunity", "potensial", "top"),
        "inventory": has("stok", "stock", "inventory", "overstock", "dead", "slow", "stockout", "cover", "replenish", "reorder"),
        "profit": has("profit", "margin", "hpp", "gross profit", "keuntungan", "laba"),
        "supplier": has("supplier", "vendor"),
        "category": has("kategori", "category", "subdept", "kel barang", "sub kel", "subkel"),
        "movement": has("mutasi", "transfer", "pembelian", "purchase", "stock in", "stock out", "movement", "opname", "adjustment", "return"),
        "anomaly": has("anomali", "anomaly", "error", "negative", "negatif", "duplicate", "duplikat", "zero price"),
        "time_detail": has("harian", "daily", "per hari", "jam", "hour", "weekday", "weekend", "minggu", "hari apa"),
    }


def build_question_context(
    bundle,
    question: str,
    as_of: pd.Timestamp,
    location: str = "IDK-ATP",
    global_filters: Optional[Dict[str, list[str]]] = None,
    previous_scope: Optional[Dict[str, Any]] = None,
    history_text: str = "",
    inventory: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Build a deterministic fact pack for the AI analyst from application data."""
    as_of = pd.Timestamp(as_of).normalize()
    search_text = f"{history_text}\n{question}".strip()
    detected = detect_entities(search_text, bundle.master)
    filters = _merge_filters(global_filters, detected)
    period = resolve_question_period(question, as_of, bundle.min_date, previous_scope=previous_scope)
    flags = _domain_flags(question)
    status_filters = _status_filters_from_question(search_text)
    pareto_filters = _pareto_filters_from_question(search_text)

    scope = {
        "period": {"start": period["start"].strftime("%Y-%m-%d"), "end": period["end"].strftime("%Y-%m-%d")},
        "period_label": period["label"],
        "filters": filters,
        "status_filters": status_filters,
        "pareto_filters": pareto_filters,
    }

    context: Dict[str, Any] = {
        "branch": location,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "data_coverage": {"start": pd.Timestamp(bundle.min_date).strftime("%Y-%m-%d"), "end": pd.Timestamp(bundle.max_date).strftime("%Y-%m-%d")},
        "question": question,
        "requested_period": {
            "start": period["requested_start"].strftime("%Y-%m-%d"),
            "end": period["requested_end"].strftime("%Y-%m-%d"),
            "effective_start": period["start"].strftime("%Y-%m-%d"),
            "effective_end": period["end"].strftime("%Y-%m-%d"),
            "data_available": period["data_available"],
            "clipped_to_data": period["clipped_to_data"],
            "label": period["label"],
        },
        "detected_entities": detected,
        "applied_filters": filters,
        "detected_inventory_status": status_filters,
        "detected_pareto_group": pareto_filters,
        "definitions": {
            "net_sales": "Penjualan dikurangi Return Penjualan.",
            "gross_profit": "Estimated Gross Profit berdasarkan estimated HPP engine aplikasi.",
            "current_stock": "Opening stock + cumulative stock in - cumulative stock out sampai tanggal as-of.",
            "core_20": "20% SKU aktif dengan revenue terbesar; tidak diasumsikan otomatis menghasilkan 80% revenue.",
            "a80": "Jumlah minimum SKU yang secara aktual membentuk 80% revenue.",
        },
    }
    tables: Dict[str, pd.DataFrame] = {}

    if not period["data_available"]:
        context["note"] = "Periode yang diminta berada di luar data yang tersedia sampai as-of date."
        return context, tables, scope

    tx_period = filter_period(bundle.tx, period["start"], period["end"])
    tx_filtered = _apply_filters(tx_period, filters)
    context["commercial_kpi"] = {k: _jsonable(v) for k, v in commercial_kpis(tx_filtered).items()}

    # Monthly trend remains useful in nearly every management question.
    tx_all_filtered = _apply_filters(bundle.tx[bundle.tx["date"].le(as_of)], filters)
    monthly = monthly_sales(tx_all_filtered)
    monthly = monthly[monthly["month"].le(as_of.to_period("M").to_timestamp())].copy()
    if bundle.targets is not None and not bundle.targets.empty and not filters:
        target_cols = [c for c in ["bulan", "lokasi", "target_omzet"] if c in bundle.targets.columns]
        tgt = bundle.targets[target_cols].copy()
        if "lokasi" in tgt.columns:
            tgt = tgt[tgt["lokasi"].astype(str).eq(str(location))]
        monthly = monthly.merge(tgt[["bulan", "target_omzet"]].rename(columns={"bulan": "month"}), on="month", how="left")
    tables["monthly_trend"] = monthly.sort_values("month").tail(24)

    # Target is a branch-level KPI; do not compare a filtered product slice with the branch target.
    if bundle.targets is not None and not bundle.targets.empty:
        target_month = period["end"].to_period("M").to_timestamp()
        target = target_status(bundle.tx, bundle.targets, target_month, min(period["end"], as_of), location)
        if target is not None:
            context["branch_target"] = {k: _jsonable(getattr(target, k)) for k in target.__dataclass_fields__}

    # Product-level sales table only when the question needs product/entity detail.
    product_cols = ["sku", "nama_barang", "supplier", "subdept", "kel_barang", "sub_kel"]
    product_detail_needed = flags["product"] or flags["target"] or flags["supplier"] or flags["category"] or flags["profit"] or bool(filters) or not any(flags.values())
    if product_detail_needed:
        product_sales = _sales_group(tx_filtered, product_cols)
        if not product_sales.empty:
            tables["product_sales"] = product_sales.head(250)

    # Sales by supplier and category are compact and useful for broad questions.
    if flags["supplier"] or not any(flags.values()):
        tables["sales_by_supplier"] = _sales_group(tx_filtered, ["supplier"]).head(100)
    if flags["category"] or not any(flags.values()):
        tables["sales_by_subdept"] = _sales_group(tx_filtered, ["subdept"]).head(100)

    # Daily/hourly/week-day patterns only when specifically useful.
    if flags["time_detail"]:
        tables["daily_sales"] = _sales_group(tx_filtered, ["date"]).sort_values("date").tail(180)
        tables["sales_by_hour"] = _sales_group(tx_filtered, ["hour"]).sort_values("hour")
        tables["sales_by_weekday"] = _sales_group(tx_filtered, ["day_name"])

    # Inventory / Pareto are central to product and action questions. For historical questions,
    # the snapshot must follow the requested period end rather than today's global as-of date.
    inventory_as_of = period["end"]
    if inventory is None or pd.Timestamp(inventory_as_of).normalize() != as_of:
        inventory = inventory_health(bundle.opening, bundle.tx, inventory_as_of, bundle.purchases)
    inv = _apply_filters(inventory, filters)
    if status_filters and "inventory_status" in inv.columns:
        inv = inv[inv["inventory_status"].isin(status_filters)].copy()
    inv_cols = [c for c in [
        "sku", "nama_barang", "supplier", "subdept", "kel_barang", "sub_kel",
        "current_stock", "current_stock_value", "current_cost", "stock_cover_days",
        "sales_qty_30d", "avg_monthly_sales", "std_monthly_sales", "days_since_last_sale",
        "inventory_status", "last_sale_date",
    ] if c in inv.columns]
    if inv_cols:
        inventory_detail_needed = flags["inventory"] or flags["product"] or flags["target"] or flags["supplier"] or flags["category"] or bool(status_filters) or bool(filters) or not any(flags.values())
        if inventory_detail_needed:
            inv_detail = inv[inv_cols].copy()
            inv_detail["abs_stock_value"] = inv_detail.get("current_stock_value", 0).abs()
            inv_detail = inv_detail.sort_values("abs_stock_value", ascending=False).drop(columns=["abs_stock_value"], errors="ignore")
            tables["inventory_detail"] = inv_detail.head(300)
        inv_summary = inv.groupby("inventory_status", as_index=False).agg(
            sku_count=("sku", "nunique"),
            stock_qty=("current_stock", "sum"),
            stock_value=("current_stock_value", "sum"),
        ).sort_values("stock_value", ascending=False)
        tables["inventory_status_summary"] = inv_summary

    pareto_needed = flags["product"] or flags["target"] or flags["inventory"] or not any(flags.values())
    if pareto_needed:
        inv_for_pareto = _apply_filters(inventory, filters)
        tx_for_pareto = _apply_filters(bundle.tx, filters)
        p = opportunity_scoring(pareto_products(tx_for_pareto, inv_for_pareto, period["start"], period["end"]), tx_for_pareto, period["end"])
        if pareto_filters and not p.empty:
            p = p[p["pareto_group"].isin(pareto_filters)].copy()
        if status_filters and not p.empty and "inventory_status" in p.columns:
            p = p[p["inventory_status"].isin(status_filters)].copy()
        if not p.empty:
            p_cols = [c for c in [
                "rank", "sku", "nama_barang", "supplier", "subdept", "pareto_group",
                "revenue", "revenue_share", "cumulative_share", "qty", "gross_profit", "gross_margin",
                "growth_30d", "current_stock", "current_stock_value", "stock_cover_days",
                "inventory_status", "opportunity_score", "recommended_action",
            ] if c in p.columns]
            tables["pareto_opportunity"] = p[p_cols].head(250)
            total_revenue = float(p["revenue"].sum())
            core = p[p["pareto_group"].eq("CORE_20")]
            opportunity = p[p["pareto_group"].eq("OPPORTUNITY")]
            context["pareto_summary"] = {
                "active_sku_in_result": int(len(p)),
                "core20_sku": int(len(core)),
                "core20_revenue_share": float(core["revenue"].sum() / total_revenue) if total_revenue else 0.0,
                "opportunity_sku": int(len(opportunity)),
                "opportunity_revenue_share": float(opportunity["revenue"].sum() / total_revenue) if total_revenue else 0.0,
                "a80_sku": int(p.get("a80_member", pd.Series(dtype=bool)).sum()) if "a80_member" in p.columns else None,
            }

    # Movement analysis (purchase / transfers / return / opname / adjustments).
    if flags["movement"]:
        movement_groups = ["movement"]
        if "mutasi" in _normalize_text(question) or "transfer" in _normalize_text(question):
            movement_groups.append("movement_partner")
        tables["movement_summary"] = _movement_summary(tx_filtered, movement_groups).head(150)
        tables["movement_by_product"] = _movement_summary(tx_filtered, ["movement", "sku", "nama_barang", "supplier"]).head(250)

    # Supplier/category productivity matrix when relevant.
    if flags["supplier"]:
        tables["supplier_productivity"] = revenue_inventory_matrix(tx_all_filtered, inv, period["start"], period["end"], "supplier").head(100)
    if flags["category"]:
        tables["category_productivity"] = revenue_inventory_matrix(tx_all_filtered, inv, period["start"], period["end"], "subdept").head(100)

    # Anomaly detail only when the question asks about data quality/errors.
    if flags["anomaly"]:
        issues = anomaly_tables(bundle.opening, bundle.tx[bundle.tx["date"].le(as_of)], inventory)
        context["anomaly_counts"] = {name: int(len(df)) for name, df in issues.items()}
        for name, df in issues.items():
            if not df.empty:
                filtered_issue = _apply_filters(df, filters)
                tables[f"anomaly_{name}"] = filtered_issue.head(200)

    # Always include a compact movement overview and inventory summary for management context.
    if "movement_summary" not in tables:
        tables["movement_summary"] = _movement_summary(tx_filtered, ["movement"]).head(50)

    context["tables"] = {name: _records(df, limit=120) for name, df in tables.items()}
    context["table_row_counts"] = {name: int(len(df)) for name, df in tables.items()}
    return context, tables, scope


def _call_openai_text(api_key: str, model: str, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        body = json.dumps({"model": model, "input": prompt}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(raw).get("error", {}).get("message") or raw
            except Exception:
                message = raw
            if exc.code == 401:
                raise RuntimeError("OpenAI API key ditolak (401). Periksa API key dan project API Anda.") from exc
            if exc.code == 429:
                raise RuntimeError(f"OpenAI quota/limit tidak tersedia (429): {message}") from exc
            raise RuntimeError(f"OpenAI API error HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Tidak dapat terhubung ke OpenAI API: {exc.reason}") from exc
        return _responses_output_text(payload)
    try:
        client = OpenAI(api_key=api_key, timeout=120.0)
        response = client.responses.create(model=model, input=prompt)
        text = (getattr(response, "output_text", None) or "").strip()
        if not text:
            raise RuntimeError("OpenAI mengembalikan respons kosong.")
        return text
    except Exception as exc:
        raise RuntimeError(f"OpenAI API request gagal: {exc}") from exc


def _gemini_text_from_payload(payload: Dict[str, Any]) -> str:
    chunks = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError("Gemini mengembalikan respons kosong.")
    return text


def _call_gemini_text(api_key: str, model: str, prompt: str) -> str:
    if _module_available("google.genai"):
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise RuntimeError("Gemini mengembalikan respons kosong.")
            return text
        except Exception as exc:
            raise RuntimeError(f"Gemini API request gagal: {exc}") from exc

    safe_model = urllib.parse.quote(model, safe="-._")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent"
    body = json.dumps(
        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            message = parsed.get("error", {}).get("message") or raw
        except Exception:
            message = raw
        if exc.code in (400, 401, 403):
            raise RuntimeError(f"Gemini API key/request ditolak ({exc.code}): {message}") from exc
        if exc.code == 429:
            raise RuntimeError(f"Gemini quota/limit tercapai (429): {message}") from exc
        raise RuntimeError(f"Gemini API error HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tidak dapat terhubung ke Gemini API: {exc.reason}") from exc
    return _gemini_text_from_payload(payload)


def generate_analyst_answer(
    api_key: str,
    model: str,
    provider: str,
    question: str,
    context: Dict[str, Any],
    history: Optional[list[dict]] = None,
    response_style: str = "Detail",
) -> str:
    provider_key = (provider or "gemini").strip().lower()
    if provider_key not in {"openai", "gemini"}:
        raise ValueError(f"AI provider tidak dikenal: {provider}")
    if not api_key:
        raise ValueError("API key belum diisi.")

    hist_lines = []
    for msg in (history or [])[-8:]:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))
        if content.strip():
            hist_lines.append(f"{role}: {content[:2500]}")
    history_text = "\n".join(hist_lines)

    style_instruction = {
        "Ringkas": "Jawab ringkas: jawaban langsung, 3-5 poin evidence, dan action jika relevan.",
        "Detail": "Jawab cukup detail dengan jawaban langsung, evidence angka, interpretasi, risiko, dan action yang relevan.",
        "Management": "Jawab seperti executive business brief: situasi, so-what, risiko, prioritas tindakan, dan target impact.",
    }.get(response_style, "Jawab cukup detail dan actionable.")

    prompt = f"""
Anda adalah Senior Business Analyst retail Mom & Baby untuk INDOKIDS. Anda menjawab pertanyaan management, buyer, inventory, dan store operations berdasarkan FACT PACK yang dihitung oleh aplikasi.

PERTANYAAN USER:
{question}

RIWAYAT CHAT TERBARU:
{history_text or '(tidak ada)'}

ATURAN DATA WAJIB:
- Gunakan HANYA angka/fakta yang tersedia di FACT PACK.
- Jangan mengarang angka, produk, supplier, target, penyebab, atau tren yang tidak didukung data.
- Jika periode yang diminta di luar coverage data, katakan jelas bahwa data tidak tersedia.
- Jika causal relationship tidak terbukti, sebut sebagai hipotesis, bukan fakta.
- Gross Profit/HPP adalah estimated apabila FACT PACK menyatakannya demikian.
- Gunakan format Rupiah seperti Rp. 1.234.567 dan persen dengan jelas.
- Jika pertanyaan meminta daftar/item/data, gunakan tabel yang tersedia sebagai evidence. Jangan mengatakan sudah melihat data yang tidak ada di FACT PACK.
- Untuk rekomendasi mengejar target, prioritaskan: protect Core products, recover stockout/low availability, grow Opportunity products yang stock-ready, lalu kurangi waste dari dead/overstock.
- Pareto bersifat dinamis; Core 20% tidak otomatis sama dengan 80% revenue.
- Berikan jawaban dalam Bahasa Indonesia kecuali user meminta bahasa lain.
- Jangan menyebut diri Anda sebagai AI.

GAYA JAWABAN:
{style_instruction}

FACT PACK (JSON):
{json.dumps(context, ensure_ascii=False, default=_jsonable)}
""".strip()

    if provider_key == "openai":
        return _call_openai_text(api_key, model, prompt)
    return _call_gemini_text(api_key, model, prompt)


def tables_to_excel(tables: Dict[str, pd.DataFrame]) -> bytes:
    """Export the fact tables used by the last AI answer into one Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        used = set()
        for idx, (name, df) in enumerate((tables or {}).items(), start=1):
            if df is None:
                continue
            clean = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or f"sheet_{idx}"
            base = clean
            n = 2
            while clean in used:
                suffix = f"_{n}"
                clean = (base[:31-len(suffix)] + suffix)
                n += 1
            used.add(clean)
            df.to_excel(writer, sheet_name=clean, index=False)
        if not used:
            pd.DataFrame({"info": ["Tidak ada tabel pendukung untuk jawaban ini."]}).to_excel(writer, sheet_name="info", index=False)
    return buf.getvalue()
