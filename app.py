from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    anomaly_tables,
    commercial_kpis,
    current_stock_snapshot,
    filter_period,
    hpp_coverage,
    inventory_health,
    monthly_sales,
    opportunity_scoring,
    pareto_products,
    revenue_inventory_matrix,
    target_status,
)
from src.cache_store import save_bundle_parquet
from src.config import SUB_KEL_FALLBACK
from src.insights import management_alerts
# AI Presentation is loaded through a compatibility layer so a partial GitHub
# update cannot crash the entire Streamlit application.
from src import ai_presentation as _ai_presentation

ai_runtime_info = _ai_presentation.ai_runtime_info

_AI_PRESENTATION_V28_REQUIRED = (
    "advanced_presentation_context",
    "build_dynamic_pptx",
    "build_recommended_slide_plan",
    "generate_ai_slide_plan",
    "generate_dynamic_presentation_content",
    "normalize_slide_plan",
    "PRESENTATION_FOCUS_OPTIONS",
    "PRESENTATION_DEPTHS",
    "SLIDE_LIBRARY",
    "recommended_focus_for_audience",
)
AI_PRESENTATION_V28_MISSING = [
    name for name in _AI_PRESENTATION_V28_REQUIRED
    if not hasattr(_ai_presentation, name)
]
AI_PRESENTATION_V28_READY = not AI_PRESENTATION_V28_MISSING

if AI_PRESENTATION_V28_READY:
    advanced_presentation_context = _ai_presentation.advanced_presentation_context
    build_dynamic_pptx = _ai_presentation.build_dynamic_pptx
    build_recommended_slide_plan = _ai_presentation.build_recommended_slide_plan
    generate_ai_slide_plan = _ai_presentation.generate_ai_slide_plan
    generate_dynamic_presentation_content = _ai_presentation.generate_dynamic_presentation_content
    normalize_slide_plan = _ai_presentation.normalize_slide_plan
    PRESENTATION_FOCUS_OPTIONS = _ai_presentation.PRESENTATION_FOCUS_OPTIONS
    PRESENTATION_DEPTHS = _ai_presentation.PRESENTATION_DEPTHS
    SLIDE_LIBRARY = _ai_presentation.SLIDE_LIBRARY
    recommended_focus_for_audience = _ai_presentation.recommended_focus_for_audience
else:
    # Safe placeholders are defined only so the module can import cleanly.
    # render_ai_presentation() exits before using them and shows a repair message.
    advanced_presentation_context = None
    build_dynamic_pptx = None
    build_recommended_slide_plan = None
    generate_ai_slide_plan = None
    generate_dynamic_presentation_content = None
    normalize_slide_plan = None
    PRESENTATION_FOCUS_OPTIONS = []
    PRESENTATION_DEPTHS = {"Standard": 14}
    SLIDE_LIBRARY = {}
    recommended_focus_for_audience = lambda audience: []
from src.ai_analyst import build_question_context, generate_analyst_answer, tables_to_excel
from src.gemini_models import GEMINI_DEFAULT_MODEL, GEMINI_MODELS, GEMINI_MODEL_LABELS
from src.io import load_raw_inputs
from src.pipeline import AnalysisBundle, build_bundle
from src.utils import file_fingerprint, pct, rupiah, style_dataframe


st.set_page_config(
    page_title="INDOKIDS Branch Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.18); padding: 13px 15px; border-radius: 13px; background: rgba(127,127,127,.035);}
.small-note {font-size: .84rem; opacity: .74;}
.hero {padding: 16px 18px; border-radius: 16px; border: 1px solid rgba(120,120,120,.18); margin-bottom: 14px;}
.hero h2 {margin: 0 0 3px 0;}
.action-critical {font-weight:700;}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4)
def process_uploads(opening_bytes: bytes, tx_bytes: bytes, purchase_bytes: Optional[bytes], target_bytes: Optional[bytes], location: str) -> AnalysisBundle:
    raw = load_raw_inputs(opening_bytes, tx_bytes, purchase_bytes, target_bytes, default_location=location)
    return build_bundle(raw)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_inventory(opening: pd.DataFrame, tx: pd.DataFrame, purchases: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    return inventory_health(opening, tx, pd.Timestamp(as_of), purchases)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_pareto(tx: pd.DataFrame, inventory: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return opportunity_scoring(pareto_products(tx, inventory, start, end), tx, end)


def df_download_button(df: pd.DataFrame, label: str, filename: str, key: str):
    data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=data, file_name=filename, mime="text/csv", key=key)


def show_table(df: pd.DataFrame, *, height=None, hide_index=True):
    """Render angka rupiah/persen secara konsisten tanpa mengubah data download."""
    styled = style_dataframe(df)
    kwargs = {"use_container_width": True, "hide_index": hide_index}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(styled, **kwargs)


def currency_axis(fig, axis="y"):
    """Format sumbu/hover nominal menggunakan prefix Rp. dan separator Indonesia."""
    fig.update_layout(separators=",.")
    if axis == "y":
        fig.update_yaxes(tickprefix="Rp. ", tickformat=",.0f")
    else:
        fig.update_xaxes(tickprefix="Rp. ", tickformat=",.0f")
    return fig


def section_title(title: str, question: str | None = None):
    st.subheader(title)
    if question:
        st.caption(question)


def render_templates():
    with st.expander("Template file input"):
        c1, c2 = st.columns(2)
        target_template = "bulan,lokasi,target_omzet\n2026-01,IDK-ATP,756695582\n"
        purchase_template = "tgl,no_faktur_beli,sku,harga_beli\n2026-01-15,BL-2601150004-H,03020320,65400\n"
        with c1:
            st.download_button("Download template Target", target_template.encode(), "template_target_cabang.csv", "text/csv")
        with c2:
            st.download_button("Download template Histori Pembelian", purchase_template.encode(), "template_histori_pembelian.csv", "text/csv")


def render_upload_page():
    st.markdown("<div class='hero'><h2>INDOKIDS Branch Performance Command Center</h2><div>Monitor · Diagnose · Act</div></div>", unsafe_allow_html=True)
    st.write("Upload **Stock Awal** dan **Kartu Stok** sebagai data wajib. Histori Pembelian dan Target bersifat opsional, tetapi diperlukan untuk analisis HPP/profitabilitas dan target chase yang lengkap.")
    render_templates()
    c1, c2 = st.columns(2)
    with c1:
        opening = st.file_uploader("1. Stock Awal (wajib)", type=["csv"], key="opening_upload")
        purchases = st.file_uploader("3. Histori Pembelian / HPP (opsional)", type=["csv"], key="purchase_upload")
    with c2:
        tx = st.file_uploader("2. Kartu Stok / Running Transaction (wajib)", type=["csv"], key="tx_upload")
        targets = st.file_uploader("4. Target Cabang (opsional)", type=["csv"], key="target_upload")
    location = st.text_input("Kode Cabang", value=st.session_state.get("location", "IDK-ATP"))
    st.session_state["location"] = location

    ready = opening is not None and tx is not None
    if not ready:
        st.info("Stock Awal dan Kartu Stok harus diupload sebelum analisis dapat dijalankan.")
        return

    if st.button("🚀 Proses Data & Jalankan Analisis", type="primary", use_container_width=True):
        opening_bytes = opening.getvalue()
        tx_bytes = tx.getvalue()
        purchase_bytes = purchases.getvalue() if purchases else None
        target_bytes = targets.getvalue() if targets else None
        fp = file_fingerprint(opening_bytes, tx_bytes, purchase_bytes, target_bytes)
        with st.spinner("Validasi, normalisasi movement, matching HPP, dan membangun cache analisis..."):
            try:
                bundle = process_uploads(opening_bytes, tx_bytes, purchase_bytes, target_bytes, location)
                st.session_state["bundle"] = bundle
                st.session_state["fingerprint"] = fp
                save_bundle_parquet(bundle, fp)
                st.success(f"Data siap. Coverage transaksi: {bundle.min_date:%d %b %Y} – {bundle.max_date:%d %b %Y}.")
                st.rerun()
            except Exception as exc:
                st.exception(exc)


def apply_product_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, values in filters.items():
        if values and col in out.columns:
            out = out[out[col].isin(values)]
    return out.copy()


def _clean_filter_dict(filters: dict) -> dict:
    """Normalize filter values so session-state comparisons stay deterministic."""
    out = {}
    for key in ["supplier", "subdept", "kel_barang", "sub_kel"]:
        values = filters.get(key, []) if isinstance(filters, dict) else []
        out[key] = [str(v) for v in values if str(v).strip()]
    return out


def _filter_summary(filters: dict) -> str:
    labels = {
        "supplier": "Supplier",
        "subdept": "Subdept",
        "kel_barang": "Kel Barang",
        "sub_kel": "Sub Kel",
    }
    parts = []
    for key, label in labels.items():
        values = filters.get(key, [])
        if values:
            preview = ", ".join(map(str, values[:2]))
            if len(values) > 2:
                preview += f" +{len(values)-2}"
            parts.append(f"{label}: {preview}")
    return " · ".join(parts) if parts else "Tidak ada filter produk (seluruh data)"


def _apply_product_filter_state():
    """Commit current sidebar widget selections as the filter used by analytics pages."""
    draft = _clean_filter_dict({
        "supplier": st.session_state.get("f_supplier", []),
        "subdept": st.session_state.get("f_subdept", []),
        "kel_barang": st.session_state.get("f_kel", []),
        "sub_kel": st.session_state.get("f_subkel", []),
    })
    st.session_state["applied_product_filters"] = draft
    st.session_state["filter_apply_counter"] = int(st.session_state.get("filter_apply_counter", 0)) + 1
    st.session_state["ask_scope_mode"] = "Ikuti Filter Produk Sidebar" if any(draft.values()) else "Seluruh Cabang"
    st.session_state.pop("business_ai_last_tables", None)
    st.session_state.pop("business_ai_last_scope", None)


def _reset_product_filter_state():
    """Clear draft and applied filters before Streamlit reruns the script."""
    for key in ["f_supplier", "f_subdept", "f_kel", "f_subkel"]:
        st.session_state[key] = []
    st.session_state["applied_product_filters"] = {
        "supplier": [], "subdept": [], "kel_barang": [], "sub_kel": []
    }
    st.session_state["ask_scope_mode"] = "Seluruh Cabang"
    st.session_state.pop("business_ai_last_tables", None)
    st.session_state.pop("business_ai_last_scope", None)


def sidebar_navigation(bundle: AnalysisBundle):
    st.sidebar.markdown("### INDOKIDS Analytics")
    st.sidebar.caption(f"Data: {bundle.min_date:%d %b %Y} – {bundle.max_date:%d %b %Y}")
    as_of = st.sidebar.date_input(
        "As of Date",
        value=bundle.max_date.date(),
        min_value=bundle.min_date.date(),
        max_value=bundle.max_date.date(),
    )
    as_of = pd.Timestamp(as_of)
    page = st.sidebar.radio(
        "Menu",
        [
            "Command Center",
            "Target Chase",
            "Pareto & Product Opportunity",
            "Sales Performance",
            "Inventory & Replenishment",
            "Profitability",
            "Category & Supplier",
            "SKU 360",
            "Ask Anything by AI",
            "AI Presentation",
            "Data & Anomaly Center",
        ],
    )

    # Applied filters are intentionally separated from widget/draft filters.
    # This makes it explicit when the user wants the application to recalculate
    # detail pages using a new filter combination.
    if "applied_product_filters" not in st.session_state:
        st.session_state["applied_product_filters"] = {
            "supplier": [], "subdept": [], "kel_barang": [], "sub_kel": []
        }

    with st.sidebar.expander("Filter Produk (halaman detail)", expanded=True):
        st.caption(
            "Pilih filter, lalu klik **Proses Data Sesuai Filter**. "
            "Command Center dan Target Chase tetap menghitung total cabang."
        )

        suppliers = st.multiselect(
            "Supplier",
            sorted(str(x) for x in bundle.master["supplier"].dropna().unique() if str(x).strip()),
            key="f_supplier",
        )
        subdepts = st.multiselect(
            "Subdept",
            sorted(str(x) for x in bundle.master["subdept"].dropna().unique() if str(x).strip()),
            key="f_subdept",
        )
        kel = st.multiselect(
            "Kel Barang",
            sorted(str(x) for x in bundle.master["kel_barang"].dropna().unique() if str(x).strip()),
            key="f_kel",
        )
        subkel_options = sorted(
            str(x) for x in bundle.master["sub_kel"].dropna().unique()
            if str(x).strip() and str(x).strip() != SUB_KEL_FALLBACK
        )
        if subkel_options:
            subkel = st.multiselect(
                "Sub Kel",
                subkel_options,
                key="f_subkel",
            )
        else:
            st.session_state["f_subkel"] = []
            subkel = []
            st.caption("Sub Kel tidak tersedia pada file cabang ini. Filter tetap bekerja sampai level Kel Barang.")

        draft_filters = _clean_filter_dict({
            "supplier": suppliers,
            "subdept": subdepts,
            "kel_barang": kel,
            "sub_kel": subkel,
        })
        applied_filters = _clean_filter_dict(st.session_state.get("applied_product_filters", {}))

        b_apply, b_reset = st.columns(2)
        b_apply.button(
            "🔄 Proses Data Sesuai Filter",
            type="primary",
            use_container_width=True,
            key="apply_product_filters_btn",
            on_click=_apply_product_filter_state,
        )
        b_reset.button(
            "Reset Filter",
            use_container_width=True,
            key="reset_product_filters_btn",
            on_click=_reset_product_filter_state,
        )

        applied_filters = _clean_filter_dict(st.session_state.get("applied_product_filters", {}))
        if draft_filters != applied_filters:
            st.warning("Pilihan filter berubah tetapi **belum diproses**. Klik tombol Proses Data Sesuai Filter.")
        st.caption("**Filter aktif:** " + _filter_summary(applied_filters))
        scoped_master = apply_product_filters(bundle.master, applied_filters)
        st.caption(f"Scope aktif: **{len(scoped_master):,} SKU** dari {len(bundle.master):,} SKU master".replace(",", "."))

    filters = _clean_filter_dict(st.session_state.get("applied_product_filters", {}))
    st.sidebar.divider()
    if st.sidebar.button("Ganti / Upload Ulang Data", use_container_width=True):
        st.session_state.pop("bundle", None)
        st.session_state.pop("fingerprint", None)
        st.session_state.pop("applied_product_filters", None)
        for key in ["f_supplier", "f_subdept", "f_kel", "f_subkel"]:
            st.session_state.pop(key, None)
        st.rerun()
    return page, as_of, filters


def month_period(as_of: pd.Timestamp):
    start = as_of.to_period("M").to_timestamp()
    return start, as_of.normalize()


def render_command_center(bundle: AnalysisBundle, as_of: pd.Timestamp, location: str):
    st.title("Command Center")
    st.caption("Pertanyaan utama: apakah cabang on-track terhadap target, dan apa risiko terbesar yang harus ditindak sekarang?")
    start, end = month_period(as_of)
    inv = cached_inventory(bundle.opening, bundle.tx, bundle.purchases, end)
    pareto = cached_pareto(bundle.tx, inv, start, end)
    target = target_status(bundle.tx, bundle.targets, start, end, location)
    kpi = commercial_kpis(filter_period(bundle.tx, start, end))

    c = st.columns(4)
    c[0].metric("Net Sales MTD", rupiah(kpi["net_sales"]))
    c[1].metric("Transactions", f"{kpi['trx_count']:,}".replace(",", "."))
    c[2].metric("Average Transaction", rupiah(kpi["atv"]))
    c[3].metric("Current Stock Value", rupiah(inv["current_stock_value"].clip(lower=0).sum()))

    c = st.columns(4)
    if target:
        c[0].metric("Target", rupiah(target.target))
        c[1].metric("Achievement", pct(target.achievement), delta=f"Pace {pct(target.pace_achievement)}")
        c[2].metric("Projected Month End", rupiah(target.projected_month_end), delta=f"Gap {rupiah(target.projected_gap)}" if target.projected_gap else "On target")
        c[3].metric("Required Daily Sales", rupiah(target.required_daily_sales))
    else:
        c[0].metric("Target", "Belum diupload")
        c[1].metric("Achievement", "-")
        c[2].metric("Projected Month End", "-")
        c[3].metric("Required Daily Sales", "-")

    section_title("Management Alerts", "Prioritas singkat yang harus dibaca sebelum membuka detail dashboard.")
    for alert in management_alerts(target, pareto, inv):
        st.warning(alert)

    left, right = st.columns([1.55, 1])
    with left:
        section_title("Sales vs Target", "Apakah perkembangan omzet bulanan sejalan dengan target?")
        monthly = monthly_sales(bundle.tx)
        if bundle.targets is not None and not bundle.targets.empty:
            mt = monthly.merge(bundle.targets[["bulan","target_omzet"]].rename(columns={"bulan":"month"}), on="month", how="left")
        else:
            mt = monthly.copy(); mt["target_omzet"] = np.nan
        mt = mt[mt["month"].le(end.to_period("M").to_timestamp())]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mt["month"], y=mt["net_sales"], mode="lines+markers", name="Net Sales"))
        if mt["target_omzet"].notna().any():
            fig.add_trace(go.Scatter(x=mt["month"], y=mt["target_omzet"], mode="lines+markers", name="Target"))
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=20,b=10), yaxis_title="Revenue", legend_title="")
        currency_axis(fig, "y")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section_title("Inventory Health", "Komposisi nilai stok berdasarkan status inventory.")
        status = inv.copy()
        status["positive_value"] = status["current_stock_value"].clip(lower=0)
        s = status.groupby("inventory_status", as_index=False)["positive_value"].sum()
        s = s[s["positive_value"].gt(0)]
        fig = px.pie(s, names="inventory_status", values="positive_value", hole=.48)
        fig.update_traces(hovertemplate="%{label}<br>Stock Value: Rp. %{value:,.0f}<extra></extra>")
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=20,b=10), legend_title="", separators=",.")
        st.plotly_chart(fig, use_container_width=True)

    section_title("Management Brief", "Ringkasan yang siap dipakai untuk meeting cabang.")
    core = pareto[pareto["pareto_group"].eq("CORE_20")]
    opportunity = pareto[pareto["pareto_group"].eq("OPPORTUNITY")]
    a80_count = int(pareto["a80_member"].sum()) if not pareto.empty else 0
    core_share = float(core["revenue"].sum() / pareto["revenue"].sum()) if not pareto.empty else 0
    cols = st.columns(4)
    cols[0].metric("Core 20% Revenue Share", pct(core_share))
    cols[1].metric("A80 SKU", f"{a80_count:,}".replace(",", "."))
    cols[2].metric("Opportunity SKU", f"{len(opportunity):,}".replace(",", "."))
    cols[3].metric("Core Stockout", f"{int((core['current_stock'] <= 0).sum()) if not core.empty else 0:,}".replace(",", "."))


def render_target_chase(bundle: AnalysisBundle, as_of: pd.Timestamp, location: str):
    st.title("Target Chase")
    st.caption("Dari gap menuju action plan: berapa kekurangan target, dan produk mana yang realistis membantu menutupnya?")
    start, end = month_period(as_of)
    target = target_status(bundle.tx, bundle.targets, start, end, location)
    if target is None:
        st.error("Target bulanan belum tersedia. Upload file Target Cabang agar halaman ini aktif penuh.")
        return
    inv = cached_inventory(bundle.opening, bundle.tx, bundle.purchases, end)
    pareto = cached_pareto(bundle.tx, inv, start, end)

    c = st.columns(5)
    c[0].metric("Target", rupiah(target.target))
    c[1].metric("Actual MTD", rupiah(target.actual))
    c[2].metric("Gap", rupiah(target.gap))
    c[3].metric("Required / Day", rupiah(target.required_daily_sales))
    c[4].metric("Projected Gap", rupiah(target.projected_gap))

    section_title("Revenue Recovery Simulator", "Uji skenario uplift pada Core dan Opportunity tanpa mengubah data sumber.")
    core_uplift = st.slider("Core 20 uplift untuk sisa bulan", 0, 50, 10, 1) / 100
    opp_uplift = st.slider("Opportunity uplift untuk sisa bulan", 0, 80, 20, 1) / 100
    recovery_pct = st.slider("Recovery potensi Core stockout", 0, 100, 70, 5) / 100

    total_rev = pareto["revenue"].sum() if not pareto.empty else 0
    core_share = pareto.loc[pareto["pareto_group"].eq("CORE_20"), "revenue"].sum() / total_rev if total_rev else 0
    opp_share = pareto.loc[pareto["pareto_group"].eq("OPPORTUNITY"), "revenue"].sum() / total_rev if total_rev else 0
    long_share = max(1 - core_share - opp_share, 0)
    remaining_base = max(target.projected_month_end - target.actual, 0)

    recent_start = max(bundle.min_date, end - pd.Timedelta(days=29))
    recent = filter_period(bundle.tx, recent_start, end)
    recent_sku = recent.groupby("sku")["net_sales_value"].sum() / max((end - recent_start).days + 1, 1)
    core_stockout = pareto[(pareto["pareto_group"].eq("CORE_20")) & pareto["current_stock"].le(0)]
    at_risk_daily = float(recent_sku.reindex(core_stockout["sku"]).fillna(0).clip(lower=0).sum())
    recovery_value = at_risk_daily * target.remaining_days * recovery_pct

    scenario_remaining = remaining_base * (
        core_share * (1 + core_uplift) + opp_share * (1 + opp_uplift) + long_share
    ) + recovery_value
    scenario_projection = target.actual + scenario_remaining
    scenario_gap = max(target.target - scenario_projection, 0)

    c = st.columns(4)
    c[0].metric("Base Projection", rupiah(target.projected_month_end))
    c[1].metric("Scenario Projection", rupiah(scenario_projection), delta=rupiah(scenario_projection - target.projected_month_end))
    c[2].metric("Scenario Gap", rupiah(scenario_gap))
    c[3].metric("Stockout Recovery", rupiah(recovery_value))

    section_title("Top Actions to Close Target", "Core dilindungi; Opportunity didorong; Long Tail dengan stok tidak sehat dikurangi.")
    action = pareto.copy()
    # Estimated incremental/at-risk revenue for action prioritisation.
    action["recent_daily_revenue"] = action["sku"].map(recent_sku).fillna(0).clip(lower=0)
    action["estimated_potential_revenue"] = np.select(
        [
            action["recommended_action"].eq("Emergency replenish"),
            action["pareto_group"].eq("CORE_20"),
            action["pareto_group"].eq("OPPORTUNITY"),
        ],
        [
            action["recent_daily_revenue"] * target.remaining_days * recovery_pct,
            action["recent_daily_revenue"] * target.remaining_days * core_uplift,
            action["recent_daily_revenue"] * target.remaining_days * opp_uplift,
        ],
        default=0.0,
    )
    priority_map = {
        "Emergency replenish": 1,
        "Protect sales / replenish": 2,
        "Push sales": 3,
        "Push / campaign candidate": 4,
        "Replenish before push": 5,
        "Maintain availability": 6,
        "Reduce buy / transfer / clearance": 7,
        "Monitor / reduce buy": 8,
        "Maintain": 9,
    }
    action["priority"] = action["recommended_action"].map(priority_map).fillna(99)
    action = action.sort_values(["priority","estimated_potential_revenue","opportunity_score","revenue"], ascending=[True,False,False,False])
    show_cols = ["sku","nama_barang","supplier","pareto_group","revenue","revenue_share","growth_30d","current_stock","stock_cover_days","inventory_status","opportunity_score","estimated_potential_revenue","recommended_action"]
    show_table(action[show_cols].head(100))
    df_download_button(action[show_cols], "Download Action List CSV", "target_chase_action_list.csv", "target_action_dl")


def render_pareto(bundle: AnalysisBundle, as_of: pd.Timestamp, filters: dict):
    st.title("Pareto & Product Opportunity")
    st.caption("Dynamic Pareto: Core 20% tetap ditandai, tetapi A80 dihitung dari kontribusi revenue aktual.")
    default_start = as_of.to_period("M").to_timestamp()
    c1, c2 = st.columns(2)
    start = pd.Timestamp(c1.date_input("Period Start", value=default_start.date(), min_value=bundle.min_date.date(), max_value=as_of.date(), key="pareto_start"))
    end = pd.Timestamp(c2.date_input("Period End", value=as_of.date(), min_value=start.date(), max_value=as_of.date(), key="pareto_end"))
    inv_full = cached_inventory(bundle.opening, bundle.tx, bundle.purchases, end)
    inv = apply_product_filters(inv_full, filters)
    tx_scope = apply_product_filters(bundle.tx, filters)
    p = opportunity_scoring(pareto_products(tx_scope, inv, start, end), tx_scope, end)
    if p.empty:
        st.info("Tidak ada revenue pada periode ini."); return
    core = p[p["pareto_group"].eq("CORE_20")]
    opp = p[p["pareto_group"].eq("OPPORTUNITY")]
    a80_n = int(p["a80_member"].sum())
    c = st.columns(4)
    c[0].metric("Active Selling SKU", f"{len(p):,}".replace(",", "."))
    c[1].metric("Core 20 SKU", f"{len(core):,}".replace(",", "."), delta=pct(core["revenue"].sum()/p["revenue"].sum()))
    c[2].metric("A80 SKU", f"{a80_n:,}".replace(",", "."), delta=pct(a80_n/len(p)))
    c[3].metric("Opportunity SKU", f"{len(opp):,}".replace(",", "."))

    left, right = st.columns(2)
    with left:
        section_title("Cumulative Revenue Pareto", "Berapa banyak SKU yang benar-benar dibutuhkan untuk membentuk 80% revenue?")
        chart = p[["rank","cumulative_share"]].copy()
        fig = px.line(chart, x="rank", y="cumulative_share")
        fig.add_hline(y=.8, line_dash="dash", annotation_text="80% revenue")
        fig.add_vline(x=len(core), line_dash="dot", annotation_text="Core 20%")
        fig.update_layout(height=390, yaxis_tickformat=".0%", margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section_title("Opportunity Map", "Score tinggi + stok siap = kandidat utama sales push.")
        plot = p.copy()
        plot["stock_cover_plot"] = plot["stock_cover_days"].clip(upper=180).fillna(180)
        fig = px.scatter(plot, x="stock_cover_plot", y="opportunity_score", size="revenue", hover_name="nama_barang", color="pareto_group", log_x=False)
        fig.update_layout(height=390, xaxis_title="Stock cover days (cap 180)", yaxis_title="Opportunity Score", margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    group_filter = st.multiselect("Pareto Group", ["CORE_20","OPPORTUNITY","LONG_TAIL"], default=["CORE_20","OPPORTUNITY"])
    table = p[p["pareto_group"].isin(group_filter)].copy()
    show_cols = ["rank","sku","nama_barang","supplier","subdept","pareto_group","revenue","revenue_share","cumulative_share","growth_30d","gross_margin","current_stock","stock_cover_days","inventory_status","opportunity_score","recommended_action"]
    show_cols = [c for c in show_cols if c in table.columns]
    show_table(table[show_cols].head(1000))
    df_download_button(table[show_cols], "Download Pareto Analysis", "pareto_product_opportunity.csv", "pareto_dl")

    section_title("Produk yang Perlu Ditingkatkan", "Daftar actionable: opportunity untuk didorong dan Core yang harus dipulihkan agar revenue tidak hilang.")
    focus_actions = ["Push sales", "Push / campaign candidate", "Replenish before push", "Protect sales / replenish", "Emergency replenish"]
    focus = p[p["recommended_action"].isin(focus_actions)].copy()
    focus["focus_type"] = np.select(
        [
            focus["recommended_action"].isin(["Emergency replenish", "Protect sales / replenish", "Replenish before push"]),
            focus["recommended_action"].isin(["Push sales", "Push / campaign candidate"]),
        ],
        ["RECOVER_AVAILABILITY", "GROW_REVENUE"],
        default="MONITOR",
    )
    focus["priority_rank"] = np.select(
        [
            focus["recommended_action"].eq("Emergency replenish"),
            focus["recommended_action"].eq("Protect sales / replenish"),
            focus["recommended_action"].eq("Push sales"),
            focus["recommended_action"].eq("Push / campaign candidate"),
            focus["recommended_action"].eq("Replenish before push"),
        ],
        [1,2,3,4,5], default=9
    )
    fc1, fc2, fc3 = st.columns(3)
    focus_statuses = fc1.multiselect("Status Inventory", sorted(x for x in focus["inventory_status"].dropna().unique()), default=sorted(x for x in focus["inventory_status"].dropna().unique()), key="focus_status")
    focus_types = fc2.multiselect("Jenis Fokus", sorted(focus["focus_type"].unique()), default=sorted(focus["focus_type"].unique()), key="focus_type")
    min_score = fc3.slider("Minimum Opportunity Score", 0, 100, 55, 5, key="focus_score")
    focus = focus[focus["inventory_status"].isin(focus_statuses) & focus["focus_type"].isin(focus_types) & focus["opportunity_score"].ge(min_score)]
    focus = focus.sort_values(["priority_rank","opportunity_score","revenue"], ascending=[True,False,False])
    focus_cols = ["focus_type","sku","nama_barang","supplier","subdept","pareto_group","revenue","revenue_share","growth_30d","gross_margin","current_stock","stock_cover_days","inventory_status","opportunity_score","recommended_action"]
    focus_cols = [c for c in focus_cols if c in focus.columns]
    show_table(focus[focus_cols].head(2000), height=520)
    df_download_button(focus[focus_cols], "Download Produk yang Perlu Ditingkatkan", "produk_perlu_ditingkatkan.csv", "focus_product_dl")


def render_sales(bundle: AnalysisBundle, as_of: pd.Timestamp, filters: dict):
    st.title("Sales Performance")
    st.caption("Diagnosa growth: apakah perubahan revenue berasal dari traffic, basket, timing, atau product mix?")
    c1, c2 = st.columns(2)
    start = pd.Timestamp(c1.date_input("Start", value=max(bundle.min_date, as_of - pd.Timedelta(days=89)).date(), min_value=bundle.min_date.date(), max_value=as_of.date(), key="sales_start"))
    end = pd.Timestamp(c2.date_input("End", value=as_of.date(), min_value=start.date(), max_value=as_of.date(), key="sales_end"))
    tx_scope = apply_product_filters(bundle.tx, filters)
    p = filter_period(tx_scope, start, end)
    k = commercial_kpis(p)
    c = st.columns(5)
    c[0].metric("Net Sales", rupiah(k["net_sales"]))
    c[1].metric("Transactions", f"{k['trx_count']:,}".replace(",", "."))
    c[2].metric("ATV", rupiah(k["atv"]))
    c[3].metric("UPT", f"{k['upt']:.2f}")
    c[4].metric("Net Qty", f"{k['net_qty']:,.0f}".replace(",", "."))

    daily = p.groupby("date", as_index=False).agg(net_sales=("net_sales_value","sum"), net_qty=("net_sales_qty","sum"))
    trx_daily = p[p["movement"].eq("SALE")].groupby("date")["kd_trx"].nunique().rename("transactions")
    daily = daily.merge(trx_daily, on="date", how="left").fillna({"transactions":0})
    section_title("Daily Sales Trend")
    fig = px.line(daily, x="date", y="net_sales")
    fig.update_layout(height=360, yaxis_title="Net Sales", margin=dict(l=10,r=10,t=10,b=10))
    currency_axis(fig, "y")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        section_title("Weekday Performance")
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wd = p.groupby("day_name", as_index=False)["net_sales_value"].sum()
        wd["day_name"] = pd.Categorical(wd["day_name"], categories=order, ordered=True)
        wd = wd.sort_values("day_name")
        fig = px.bar(wd, x="day_name", y="net_sales_value")
        fig.update_layout(height=340, xaxis_title="", yaxis_title="Net Sales", margin=dict(l=10,r=10,t=10,b=10))
        currency_axis(fig, "y")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section_title("Hourly Sales")
        if "time_available" in p.columns and not p["time_available"].fillna(False).any():
            st.info(
                "Jam transaksi tidak tersedia pada format file ini. Tanggal transaksi berhasil direkonstruksi dari kode transaksi, "
                "tetapi aplikasi sengaja tidak mengarang jam agar analisis hourly tetap akurat."
            )
        else:
            hourly_source = p[p["time_available"].fillna(True)] if "time_available" in p.columns else p
            hr = hourly_source.groupby("hour", as_index=False)["net_sales_value"].sum()
            fig = px.bar(hr, x="hour", y="net_sales_value")
            fig.update_layout(height=340, xaxis_title="Hour", yaxis_title="Net Sales", margin=dict(l=10,r=10,t=10,b=10))
            currency_axis(fig, "y")
            st.plotly_chart(fig, use_container_width=True)


def render_inventory(bundle: AnalysisBundle, as_of: pd.Timestamp, filters: dict):
    st.title("Inventory & Replenishment")
    st.caption("Protect winners, detect stockout risk, dan hentikan modal tertahan pada stok yang tidak produktif.")
    inv = apply_product_filters(cached_inventory(bundle.opening, bundle.tx, bundle.purchases, as_of), filters)
    c = st.columns(5)
    c[0].metric("Current Stock Qty", f"{inv['current_stock'].sum():,.0f}".replace(",", "."))
    c[1].metric("Current Stock Value", rupiah(inv["current_stock_value"].clip(lower=0).sum()))
    c[2].metric("Stockout w/ Demand", f"{(inv['inventory_status']=='STOCKOUT').sum():,}".replace(",", "."))
    c[3].metric("Overstock", f"{(inv['inventory_status']=='OVERSTOCK').sum():,}".replace(",", "."))
    c[4].metric("Negative", f"{(inv['inventory_status']=='NEGATIVE').sum():,}".replace(",", "."))

    status = inv.groupby("inventory_status", as_index=False).agg(sku=("sku","nunique"), stock_value=("current_stock_value", lambda s: s.clip(lower=0).sum()))
    left, right = st.columns(2)
    with left:
        section_title("SKU by Inventory Status")
        fig = px.bar(status.sort_values("sku", ascending=False), x="inventory_status", y="sku")
        fig.update_layout(height=360, xaxis_title="", yaxis_title="SKU", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section_title("Stock Value by Status")
        fig = px.bar(status.sort_values("stock_value", ascending=False), x="inventory_status", y="stock_value")
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Stock Value", margin=dict(l=10,r=10,t=10,b=10))
        currency_axis(fig, "y")
        st.plotly_chart(fig, use_container_width=True)

    statuses = st.multiselect("Inventory Status", sorted(inv["inventory_status"].unique()), default=[x for x in ["STOCKOUT","SLOW","DEAD","OVERSTOCK","NEGATIVE"] if x in inv["inventory_status"].unique()])
    table = inv[inv["inventory_status"].isin(statuses)].copy()
    table = table.sort_values(["inventory_status","current_stock_value"], ascending=[True,False])
    cols = ["sku","nama_barang","supplier","subdept","current_stock","current_cost","current_stock_value","sales_qty_30d","stock_cover_days","avg_monthly_sales","std_monthly_sales","slow_threshold","dead_threshold","overstock_threshold","days_since_last_sale","inventory_status"]
    show_table(table[cols].head(2000))
    df_download_button(table[cols], "Download Inventory Action Data", "inventory_action_data.csv", "inventory_dl")


def render_profitability(bundle: AnalysisBundle, as_of: pd.Timestamp, filters: dict):
    st.title("Profitability")
    st.caption("Estimated HPP menggunakan cost resolution engine; bukan pengganti HPP akuntansi FIFO/moving average.")
    if "hpp_source" not in bundle.tx.columns:
        st.error("HPP engine tidak tersedia."); return
    start = as_of.to_period("M").to_timestamp()
    tx_scope = apply_product_filters(bundle.tx, filters)
    p = filter_period(tx_scope, start, as_of)
    k = commercial_kpis(p)
    c = st.columns(4)
    c[0].metric("Net Sales", rupiah(k["net_sales"]))
    c[1].metric("Estimated HPP", rupiah(k["net_hpp"]))
    c[2].metric("Estimated Gross Profit", rupiah(k["gross_profit"]))
    c[3].metric("Gross Margin", pct(k["gross_margin"]))

    cov = hpp_coverage(p)
    section_title("HPP Coverage")
    if not cov.empty:
        fig = px.bar(cov, x="hpp_source", y="lines")
        fig.update_layout(height=330, xaxis_title="Source", yaxis_title="Sales Lines", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        show_table(cov)

    commercial = p[p["movement"].isin(["SALE","SALES_RETURN"])].copy()
    by_supplier = commercial.groupby("supplier", as_index=False).agg(net_sales=("net_sales_value","sum"), hpp=("net_hpp","sum"))
    by_supplier["gross_profit"] = by_supplier["net_sales"] - by_supplier["hpp"]
    by_supplier["gross_margin"] = by_supplier["gross_profit"] / by_supplier["net_sales"].replace(0,np.nan)
    by_supplier = by_supplier.sort_values("gross_profit", ascending=False)
    section_title("Top Supplier by Estimated Gross Profit")
    show_table(by_supplier.head(100))


def render_category_supplier(bundle: AnalysisBundle, as_of: pd.Timestamp, filters: dict):
    st.title("Category & Supplier")
    st.caption("Bandingkan revenue share dengan inventory share untuk menemukan kategori produktif, understock, dan capital-heavy.")
    start = as_of.to_period("M").to_timestamp()
    inv = apply_product_filters(cached_inventory(bundle.opening, bundle.tx, bundle.purchases, as_of), filters)
    tx_scope = apply_product_filters(bundle.tx, filters)
    dimensions = ["supplier", "subdept", "kel_barang"]
    if bundle.master["sub_kel"].astype(str).ne(SUB_KEL_FALLBACK).any():
        dimensions.append("sub_kel")
    dim = st.selectbox("Dimension", dimensions)
    matrix = revenue_inventory_matrix(tx_scope, inv, start, as_of, dim)
    matrix["interpretation"] = np.select(
        [matrix["productivity_index"].ge(1.5), matrix["productivity_index"].le(.65)],
        ["Productive / potential understock", "Capital heavy"],
        default="Balanced"
    )
    c1, c2 = st.columns([1.35,1])
    with c1:
        fig = px.scatter(matrix, x="inventory_share", y="revenue_share", size="revenue", hover_name=dim, color="interpretation")
        maxv = max(matrix["inventory_share"].max(), matrix["revenue_share"].max(), .01)
        fig.add_shape(type="line", x0=0,y0=0,x1=maxv,y1=maxv,line=dict(dash="dash"))
        fig.update_layout(height=430, xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="Inventory Share", yaxis_title="Revenue Share")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        show_table(matrix[[dim,"revenue","inventory_value","revenue_share","inventory_share","productivity_index","interpretation"]].head(100), height=430)


def render_sku360(bundle: AnalysisBundle, as_of: pd.Timestamp, filters: dict):
    st.title("SKU 360")
    st.caption("Satu layar untuk memahami kontribusi, histori movement, cost, dan kondisi inventory sebuah SKU.")
    inv = apply_product_filters(cached_inventory(bundle.opening, bundle.tx, bundle.purchases, as_of), filters)
    if inv.empty:
        st.warning("Tidak ada SKU pada kombinasi filter saat ini.")
        return

    status_options = sorted(x for x in inv["inventory_status"].dropna().unique() if str(x).strip())
    selected_status = st.multiselect(
        "Filter Status",
        status_options,
        default=status_options,
        help="Gunakan filter ini untuk fokus pada STOCKOUT, SLOW, DEAD, OVERSTOCK, NEGATIVE, NORMAL, dan status inventory lainnya.",
        key="sku360_status",
    )
    inv = inv[inv["inventory_status"].isin(selected_status)].copy()
    if inv.empty:
        st.warning("Tidak ada SKU dengan Status yang dipilih.")
        return

    labels = inv[["sku","nama_barang","supplier","subdept","kel_barang","sub_kel","inventory_status"]].drop_duplicates("sku").copy()
    labels["label"] = labels["sku"].astype(str) + " — " + labels["nama_barang"].fillna("").astype(str) + " [" + labels["inventory_status"].astype(str) + "]"
    choice = st.selectbox("Pilih SKU", labels.sort_values(["inventory_status","label"])["label"].tolist())
    sku = choice.split(" — ",1)[0]
    meta = inv[inv["sku"].eq(sku)]
    if meta.empty:
        st.warning("SKU tidak ditemukan pada snapshot.")
        return
    m = meta.iloc[0]
    hierarchy = f"{m['subdept']} / {m['kel_barang']}"
    if str(m.get("sub_kel", "")).strip() not in {"", SUB_KEL_FALLBACK}:
        hierarchy += f" / {m['sub_kel']}"
    st.markdown(f"### {m['nama_barang']}  \n`{sku}` · {m['supplier']} · {hierarchy}")
    txs = bundle.tx[(bundle.tx["sku"].eq(sku)) & bundle.tx["date"].le(as_of)].copy()
    k = commercial_kpis(txs)
    c = st.columns(5)
    c[0].metric("Revenue", rupiah(k["net_sales"]))
    c[1].metric("Gross Profit", rupiah(k["gross_profit"]))
    c[2].metric("Current Stock", f"{m['current_stock']:,.0f}".replace(",", "."))
    c[3].metric("Stock Cover", f"{m['stock_cover_days']:.1f} hari" if pd.notna(m['stock_cover_days']) else "-")
    c[4].metric("Status", str(m["inventory_status"]))

    opening_qty = float(bundle.opening.loc[bundle.opening["sku"].eq(sku), "saldo_awal"].sum())
    movement_daily = txs.groupby("date", as_index=False).agg(stock_in=("stock_in","sum"), stock_out=("stock_out","sum"), net_sales=("net_sales_value","sum"))
    movement_daily["stock_balance"] = opening_qty + (movement_daily["stock_in"] - movement_daily["stock_out"]).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=movement_daily["date"], y=movement_daily["stock_balance"], mode="lines", name="Stock Balance"))
    fig.update_layout(height=370, yaxis_title="Qty", margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig, use_container_width=True)

    cols = ["tgl","kd_trx","movement","stock_in","stock_out","harga","subtotal","hpp_unit","hpp_source","keterangan"]
    cols = [c for c in cols if c in txs.columns]
    show_table(txs.sort_values("tgl", ascending=False)[cols].head(1000))



def _streamlit_secret(name: str) -> str:
    """Read an optional Streamlit secret without failing on local development."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value or "").strip()


def render_ask_ai(bundle: AnalysisBundle, as_of: pd.Timestamp, location: str, filters: dict):
    st.title("Ask Anything by AI")
    st.caption(
        "Tanyakan apa pun tentang performa cabang. Aplikasi lebih dulu menghitung fact pack dari data yang diupload; "
        "OpenAI/Gemini kemudian bertugas membaca fakta tersebut sebagai Business Analyst berpengalaman."
    )

    with st.expander("AI Analyst Settings", expanded=True):
        c0, c1, c2 = st.columns([1.05, 1.2, 1.0])
        provider_label = c0.selectbox(
            "AI Provider",
            ["Google Gemini", "OpenAI"],
            index=0,
            key="ask_ai_provider",
        )
        provider = "gemini" if provider_label == "Google Gemini" else "openai"
        if provider == "gemini":
            if st.session_state.get("ask_gemini_model") not in (None, *GEMINI_MODELS):
                st.session_state["ask_gemini_model"] = GEMINI_DEFAULT_MODEL
            model = c1.selectbox(
                "Model",
                GEMINI_MODELS,
                index=GEMINI_MODELS.index(GEMINI_DEFAULT_MODEL),
                format_func=lambda x: GEMINI_MODEL_LABELS.get(x, x),
                help="Default: Gemini 3.7 Flash. Auto-fallback hanya berjalan bila model tidak tersedia/deprecated.",
                key="ask_gemini_model",
            )
            secret_key = _streamlit_secret("GEMINI_API_KEY")
            if secret_key:
                api_key = secret_key
                c2.success("Gemini API Key: Streamlit Secrets")
            else:
                api_key = c2.text_input(
                    "Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    key="ask_gemini_api_key",
                )
        else:
            model = c1.selectbox("Model", ["gpt-5.6"], index=0, key="ask_openai_model")
            secret_key = _streamlit_secret("OPENAI_API_KEY")
            if secret_key:
                api_key = secret_key
                c2.success("OpenAI API Key: Streamlit Secrets")
            else:
                api_key = c2.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    key="ask_openai_api_key",
                )

        c3, c4, c5 = st.columns([1.1, 1.1, 1.0])
        active_filter_exists = any(bool(v) for v in filters.values())
        if "ask_scope_mode" not in st.session_state:
            st.session_state["ask_scope_mode"] = "Ikuti Filter Produk Sidebar" if active_filter_exists else "Seluruh Cabang"
        scope_mode = c3.selectbox(
            "Data Scope",
            ["Seluruh Cabang", "Ikuti Filter Produk Sidebar"],
            help="Setelah tombol Proses Data Sesuai Filter diklik, Ask AI otomatis mengikuti filter aktif. Target tetap branch-level.",
            key="ask_scope_mode",
        )
        response_style = c4.selectbox(
            "Gaya Jawaban",
            ["Detail", "Ringkas", "Management"],
            index=0,
            key="ask_response_style",
        )
        c5.caption(f"As of Date\n\n**{pd.Timestamp(as_of):%d %b %Y}**")
        if provider == "gemini":
            st.caption("Auto-fallback Gemini: 3.7 Flash → 3.6 Flash → 3.5 Flash → 3.5 Flash-Lite. Fallback hanya dipakai jika model tidak tersedia, bukan untuk error API key/quota.")

        runtime = ai_runtime_info()
        if provider == "gemini":
            transport = "Google GenAI SDK" if runtime["gemini_sdk"] else "HTTPS fallback"
        else:
            transport = "OpenAI Python SDK" if runtime["openai_sdk"] else "HTTPS fallback"
        st.caption(f"Runtime: {transport} · API key tidak disimpan ke dataset/cache aplikasi.")
        if scope_mode == "Ikuti Filter Produk Sidebar":
            st.success("Ask AI memakai filter aktif: " + _filter_summary(filters))
        else:
            st.caption("Ask AI memakai seluruh data cabang (filter produk tidak diterapkan).")

    st.info(
        "Contoh pertanyaan: **Berapa gap target Agustus dan produk apa yang paling realistis didorong?** · "
        "**Supplier mana revenue terbesar tetapi inventory productivity rendah?** · "
        "**Tampilkan item OVERSTOCK dari supplier tertentu** · "
        "**Berapa nilai mutasi keluar bulan Juli dan ke lokasi mana paling besar?**"
    )

    # Chat state.
    if "business_ai_messages" not in st.session_state:
        st.session_state["business_ai_messages"] = []
    messages = st.session_state["business_ai_messages"]

    top_left, top_right = st.columns([1, 4])
    if top_left.button("🧹 Clear Chat", use_container_width=True, key="clear_business_ai"):
        st.session_state["business_ai_messages"] = []
        st.session_state.pop("business_ai_last_tables", None)
        st.session_state.pop("business_ai_last_scope", None)
        st.rerun()
    top_right.caption(
        "AI hanya menerima data hasil kalkulasi/aggregasi yang relevan dengan pertanyaan. Raw transaction tidak dikirim langsung ke provider AI."
    )

    for msg in messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

    question = st.chat_input("Tanyakan sesuatu tentang data cabang INDOKIDS...", disabled=not bool(str(api_key or "").strip()))
    if not question:
        last_tables = st.session_state.get("business_ai_last_tables")
        if last_tables:
            with st.expander("Data pendukung jawaban terakhir", expanded=False):
                for name, records in last_tables.items():
                    df = pd.DataFrame(records)
                    st.markdown(f"**{name.replace('_', ' ').title()}**")
                    show_table(df.head(300), height=320)
                payload = tables_to_excel({k: pd.DataFrame(v) for k, v in last_tables.items()})
                st.download_button(
                    "⬇️ Download data pendukung (.xlsx)",
                    data=payload,
                    file_name=f"INDOKIDS_{location}_{pd.Timestamp(as_of):%Y%m%d}_AI_supporting_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_ai_support_last",
                )
        return

    # Display and persist user question.
    with st.chat_message("user"):
        st.markdown(question)
    messages.append({"role": "user", "content": question})

    use_filters = filters if scope_mode == "Ikuti Filter Produk Sidebar" else {}
    history_text = "\n".join(
        f"{m.get('role','user')}: {m.get('content','')}" for m in messages[-6:-1]
    )
    previous_scope = st.session_state.get("business_ai_last_scope")
    inv = cached_inventory(bundle.opening, bundle.tx, bundle.purchases, as_of)

    with st.chat_message("assistant"):
        with st.spinner(f"{provider_label} sedang membaca data dan menyusun analisis..."):
            try:
                context, source_tables, scope = build_question_context(
                    bundle,
                    question,
                    as_of,
                    location=location,
                    global_filters=use_filters,
                    previous_scope=previous_scope,
                    history_text=history_text,
                    inventory=inv,
                )
                answer = generate_analyst_answer(
                    str(api_key).strip(),
                    model,
                    provider,
                    question,
                    context,
                    history=messages[:-1],
                    response_style=response_style,
                )
                st.markdown(answer)
                messages.append({"role": "assistant", "content": answer})
                st.session_state["business_ai_messages"] = messages
                st.session_state["business_ai_last_scope"] = scope
                serial_tables = {
                    name: df.head(300).replace({np.nan: None}).to_dict("records")
                    for name, df in source_tables.items() if df is not None and not df.empty
                }
                st.session_state["business_ai_last_tables"] = serial_tables
            except Exception as exc:
                message = str(exc)
                if "429" in message:
                    st.error(
                        f"{provider_label} tidak dapat memproses request karena quota/limit API. "
                        "Coba provider lain atau periksa quota/billing provider."
                    )
                    st.caption(message)
                elif "401" in message or "403" in message:
                    st.error(f"{provider_label} menolak API key/request. Periksa key dan akses model.")
                    st.caption(message)
                else:
                    st.error(f"AI Analyst gagal memproses pertanyaan: {message}")
                return

    last_tables = st.session_state.get("business_ai_last_tables")
    if last_tables:
        with st.expander("Data pendukung yang digunakan untuk jawaban ini", expanded=False):
            for name, records in last_tables.items():
                df = pd.DataFrame(records)
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                show_table(df.head(300), height=320)
            payload = tables_to_excel({k: pd.DataFrame(v) for k, v in last_tables.items()})
            st.download_button(
                "⬇️ Download data pendukung (.xlsx)",
                data=payload,
                file_name=f"INDOKIDS_{location}_{pd.Timestamp(as_of):%Y%m%d}_AI_supporting_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_ai_support_new",
            )

def render_ai_presentation(bundle: AnalysisBundle, as_of: pd.Timestamp, location: str):
    st.title("AI Presentation Studio")

    if not AI_PRESENTATION_V28_READY:
        st.error(
            "Module AI Presentation belum sinkron dengan app.py V2.8. "
            "Aplikasi utama tetap dapat digunakan, tetapi AI Presentation Studio dinonaktifkan sementara."
        )
        st.warning(
            "Replace **dua file sekaligus** dari hotfix V2.8.1: `app.py` dan `src/ai_presentation.py`, "
            "lalu commit/push ke GitHub dan reboot Streamlit."
        )
        st.code(
            "Missing V2.8 symbols:\n- " + "\n- ".join(AI_PRESENTATION_V28_MISSING),
            language="text",
        )
        return

    st.caption(
        "Bangun presentasi management yang dinamis: tentukan objective, fokus, kedalaman, dan poin wajib; "
        "AI menyusun slide plan dan insight, sedangkan angka/chart/table tetap berasal dari engine aplikasi."
    )

    with st.expander("1. AI Provider & Audience", expanded=True):
        provider_label = st.selectbox(
            "AI Provider",
            ["Google Gemini", "OpenAI"],
            index=0,
            help="Provider hanya menyusun struktur dan narasi. Angka PowerPoint selalu berasal dari analytics engine lokal.",
        )
        provider = "gemini" if provider_label == "Google Gemini" else "openai"

        if provider == "gemini":
            if st.session_state.get("gemini_model") not in (None, *GEMINI_MODELS):
                st.session_state["gemini_model"] = GEMINI_DEFAULT_MODEL
            secret_key = _streamlit_secret("GEMINI_API_KEY")
            if secret_key:
                api_key = secret_key
                st.success("Gemini API Key dimuat dari Streamlit Secrets.")
            else:
                api_key = st.text_input(
                    "Gemini API Key", type="password", value="", placeholder="AIza...",
                    help="Key hanya dipakai untuk request AI dan tidak disimpan ke dataset/cache.", key="presentation_gemini_api_key",
                )
            c1, c2, c3 = st.columns(3)
            model = c1.selectbox(
                "Model", GEMINI_MODELS,
                index=GEMINI_MODELS.index(GEMINI_DEFAULT_MODEL),
                format_func=lambda x: GEMINI_MODEL_LABELS.get(x, x),
                key="presentation_gemini_model",
            )
        else:
            secret_key = _streamlit_secret("OPENAI_API_KEY")
            if secret_key:
                api_key = secret_key
                st.success("OpenAI API Key dimuat dari Streamlit Secrets.")
            else:
                api_key = st.text_input(
                    "OpenAI API Key", type="password", value="", placeholder="sk-...",
                    help="Key hanya dipakai untuk request AI dan tidak disimpan ke dataset/cache.", key="presentation_openai_api_key",
                )
            c1, c2, c3 = st.columns(3)
            model = c1.selectbox("Model", ["gpt-5.6"], index=0, key="presentation_openai_model")

        audience = c2.selectbox(
            "Audience",
            ["Management / Owner", "Buyer & Inventory", "Store Operations"],
            index=0,
            key="presentation_audience",
            help="Audience mengubah penekanan insight dan rekomendasi slide.",
        )
        language = c3.selectbox(
            "Bahasa Presentasi", ["Bahasa Indonesia", "English"], index=0, key="presentation_language"
        )

        runtime = ai_runtime_info()
        transport = (
            "Google GenAI SDK" if provider == "gemini" and runtime["gemini_sdk"] else
            "OpenAI Python SDK" if provider == "openai" and runtime["openai_sdk"] else
            "HTTPS fallback"
        )
        st.caption(f"Runtime AI: {transport} · PowerPoint engine: {'siap' if runtime['python_pptx'] else 'belum terinstall'}")

    context = advanced_presentation_context(bundle, as_of, location)
    target = context.get("target", {}) or {}
    m = st.columns(4)
    m[0].metric("Net Sales MTD", rupiah(context["kpi"].get("net_sales")))
    m[1].metric("Target", rupiah(target.get("target")) if target else "-")
    m[2].metric("Projected Gap", rupiah(target.get("projected_gap")) if target else "-")
    m[3].metric("Core 20 Share", pct(context.get("pareto", {}).get("core_share")))

    with st.expander("2. Presentation Brief", expanded=True):
        if "presentation_objective" not in st.session_state:
            st.session_state["presentation_objective"] = (
                "Evaluasi performa cabang, menjelaskan gap terhadap target, dan menentukan tindakan paling realistis "
                "untuk meningkatkan revenue tanpa memperburuk margin maupun kesehatan inventory."
            )
        objective = st.text_area(
            "Tujuan Presentasi",
            height=92,
            key="presentation_objective",
        )
        depth = st.selectbox(
            "Kedalaman Presentasi",
            list(PRESENTATION_DEPTHS.keys()),
            index=1,
            key="presentation_depth",
            help="Executive cocok untuk meeting singkat; Deep Dive cocok untuk evaluasi buyer/inventory yang lebih detail.",
        )
        default_focus = recommended_focus_for_audience(audience)
        focus_areas = st.multiselect(
            "Fokus / Topik yang Ingin Dibahas",
            PRESENTATION_FOCUS_OPTIONS,
            default=default_focus,
            key=f"presentation_focus_{audience}",
            help="Topik yang dipilih menjadi kandidat slide. AI tetap akan menghindari slide yang tidak didukung data.",
        )
        if "presentation_additional_points" not in st.session_state:
            st.session_state["presentation_additional_points"] = ""
        additional_points = st.text_area(
            "Poin Tambahan yang Wajib Dibahas",
            placeholder=(
                "Contoh:\n"
                "1. Jelaskan penyebab cabang belum mencapai target berdasarkan data yang tersedia.\n"
                "2. Fokuskan 20% Core Product dan Opportunity Product yang paling realistis didorong.\n"
                "3. Cari produk stockout dengan histori revenue tinggi.\n"
                "4. Evaluasi supplier dengan inventory share besar tetapi revenue share rendah.\n"
                "5. Berikan maksimal 10 tindakan konkret untuk 30 hari ke depan."
            ),
            height=150,
            key="presentation_additional_points",
        )
        st.caption(
            "Brief ini masuk ke prompt AI sebagai **mandatory presentation brief**, tetapi AI tetap dilarang mengarang fakta yang tidak ada di context aplikasi."
        )

    st.subheader("3. Generate & Review Slide Plan")
    st.caption(
        "AI menyusun urutan cerita terlebih dahulu. Anda dapat mengedit judul, objective, emphasis, urutan, menghapus slide, atau menambah slide sebelum PowerPoint final dibuat."
    )
    p1, p2, p3 = st.columns([1.6, 1.6, 1])
    generate_plan = p1.button(
        "🧠 Generate AI Slide Plan",
        type="primary",
        use_container_width=True,
        disabled=not bool(str(api_key).strip()),
    )
    local_plan = p2.button("📋 Gunakan Recommended Plan", use_container_width=True)
    reset_plan = p3.button("Reset", use_container_width=True)

    if reset_plan:
        for k in ["presentation_slide_plan", "presentation_dynamic_ai", "presentation_dynamic_pptx"]:
            st.session_state.pop(k, None)
        st.rerun()

    if generate_plan:
        with st.spinner(f"{provider_label} sedang menyusun arsitektur presentasi..."):
            try:
                plan = generate_ai_slide_plan(
                    str(api_key).strip(), model, context,
                    audience=audience, language=language, objective=objective,
                    focus_areas=focus_areas, additional_points=additional_points,
                    depth=depth, provider=provider,
                )
                st.session_state["presentation_slide_plan"] = plan
                st.session_state.pop("presentation_dynamic_ai", None)
                st.session_state.pop("presentation_dynamic_pptx", None)
                st.success(f"Slide plan dibuat: {len(plan)} slide. Silakan review/edit sebelum generate final deck.")
            except Exception as exc:
                st.error(f"Gagal membuat AI Slide Plan: {exc}")

    if local_plan:
        plan = build_recommended_slide_plan(focus_areas, depth, audience)
        st.session_state["presentation_slide_plan"] = plan
        st.session_state.pop("presentation_dynamic_ai", None)
        st.session_state.pop("presentation_dynamic_pptx", None)
        st.success(f"Recommended slide plan dibuat: {len(plan)} slide.")

    plan = st.session_state.get("presentation_slide_plan")
    edited_plan = None
    if plan:
        plan_df = pd.DataFrame(plan)
        edited_df = st.data_editor(
            plan_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="presentation_plan_editor",
            column_config={
                "include": st.column_config.CheckboxColumn("Include", default=True, width="small"),
                "order": st.column_config.NumberColumn("Order", min_value=1, step=1, width="small"),
                "slide_type": st.column_config.SelectboxColumn(
                    "Slide Type", options=list(SLIDE_LIBRARY.keys()), required=True, width="medium"
                ),
                "title": st.column_config.TextColumn("Slide Title", width="large"),
                "objective": st.column_config.TextColumn("Objective", width="large"),
                "emphasis": st.column_config.TextColumn("Emphasis / Notes", width="large"),
            },
        )
        edited_plan = normalize_slide_plan(edited_df.to_dict("records"), depth)
        st.caption(f"Final plan saat ini: **{len(edited_plan)} slide** · maksimal {PRESENTATION_DEPTHS.get(depth, 14)} slide sesuai depth.")
        with st.expander("Preview urutan slide", expanded=False):
            preview = pd.DataFrame(edited_plan)[["order", "slide_type", "title", "objective", "emphasis"]]
            show_table(preview)

    st.subheader("4. Generate Final AI Insights & PowerPoint")
    st.caption(
        "Tahap ini meminta AI menulis insight spesifik untuk setiap slide yang sudah Anda approve. Chart, KPI, tabel, dan nominal tetap dibangun dari fact-pack aplikasi."
    )
    final_button = st.button(
        f"✨ Generate Final Deck with {provider_label}",
        type="primary",
        use_container_width=True,
        disabled=not (bool(str(api_key).strip()) and bool(edited_plan)),
    )
    if final_button:
        with st.spinner("AI sedang menyusun slide-specific insight lalu aplikasi membangun PowerPoint dinamis..."):
            try:
                ai = generate_dynamic_presentation_content(
                    str(api_key).strip(), model, context, edited_plan,
                    audience=audience, language=language, objective=objective,
                    additional_points=additional_points, provider=provider,
                )
                pptx_bytes = build_dynamic_pptx(
                    context, ai, edited_plan, branch=location, as_of=as_of,
                    audience=audience, objective=objective,
                )
                st.session_state["presentation_slide_plan"] = edited_plan
                st.session_state["presentation_dynamic_ai"] = ai
                st.session_state["presentation_dynamic_pptx"] = pptx_bytes
                st.session_state["presentation_dynamic_provider"] = provider_label
                st.success(f"Dynamic presentation berhasil dibuat menggunakan {provider_label}.")
            except Exception as exc:
                message = str(exc); lower = message.lower()
                if "429" in lower or "quota" in lower or "insufficient_quota" in lower:
                    st.error(f"{provider_label} tidak dapat memproses request karena quota/limit API.")
                    st.caption(message)
                elif "401" in lower or "403" in lower or "api key" in lower:
                    st.error(f"{provider_label} menolak API key/request. Periksa key dan akses model.")
                    st.caption(message)
                else:
                    st.error(f"Gagal membuat dynamic AI presentation: {message}")

    ai = st.session_state.get("presentation_dynamic_ai")
    pptx_bytes = st.session_state.get("presentation_dynamic_pptx")
    last_provider = st.session_state.get("presentation_dynamic_provider")
    if ai:
        section_title(f"AI Presentation Readout{f' · {last_provider}' if last_provider else ''}")
        takeaway = ai.get("executive_takeaway")
        if takeaway:
            st.info(takeaway)
        slide_rows = []
        for item in ai.get("slides", []):
            slide_rows.append({
                "slide_type": item.get("slide_type"),
                "title": item.get("title"),
                "headline": item.get("headline"),
                "risk_note": item.get("risk_note"),
                "action_note": item.get("action_note"),
            })
        if slide_rows:
            with st.expander("Preview AI headline per slide", expanded=False):
                show_table(pd.DataFrame(slide_rows))
        actions = pd.DataFrame(ai.get("recommended_actions", []))
        if not actions.empty:
            section_title("30-Day Recommended Actions")
            show_table(actions)

    if pptx_bytes:
        st.download_button(
            "⬇️ Download Dynamic PowerPoint (.pptx)",
            data=pptx_bytes,
            file_name=f"INDOKIDS_{location}_{pd.Timestamp(as_of):%Y%m%d}_AI_Presentation_Studio.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )

    with st.expander("Prompt governance & data privacy", expanded=False):
        st.markdown(
            """
**AI menerima:** KPI agregat, target & forecast, tren bulanan/harian, Pareto, product opportunity, inventory health/capital, supplier/category productivity, transfer/purchase summary, profitability, dan anomaly counts.  
**AI tidak menerima:** seluruh raw transaction secara langsung.  
**Governance:** AI dilarang mengarang angka/kausalitas; jika penyebab tidak terbukti, harus diposisikan sebagai hipotesis atau investigasi yang direkomendasikan.  
**PowerPoint builder:** chart, tabel, KPI, dan nominal dibuat oleh aplikasi dari analytics context; AI fokus pada story, headline, interpretation, dan action plan.
            """
        )

def render_data_anomaly(bundle: AnalysisBundle, as_of: pd.Timestamp):
    st.title("Data & Anomaly Center")
    st.caption("Pisahkan masalah data dari masalah bisnis agar rekomendasi tidak dibangun di atas data yang keliru.")
    inv = cached_inventory(bundle.opening, bundle.tx, bundle.purchases, as_of)
    tx_for_anomaly = bundle.tx[bundle.tx["date"].le(as_of) | bundle.tx["date"].isna()]
    issues = anomaly_tables(bundle.opening, tx_for_anomaly, inv)
    c = st.columns(4)
    c[0].metric("Opening SKU", f"{bundle.opening['sku'].nunique():,}".replace(",", "."))
    c[1].metric("Transaction Rows", f"{len(bundle.tx):,}".replace(",", "."))
    c[2].metric("Active Transaction SKU", f"{bundle.tx['sku'].nunique():,}".replace(",", "."))
    c[3].metric("Data Coverage", f"{bundle.min_date:%d %b} – {bundle.max_date:%d %b %Y}")

    subkel_real = bundle.master["sub_kel"].astype(str).ne(SUB_KEL_FALLBACK).any()
    reconstructed = int(bundle.tx.get("date_parse_status", pd.Series(dtype=str)).eq("KD_TRX_DATE").sum()) if "date_parse_status" in bundle.tx.columns else 0
    if not subkel_real:
        st.info("Format cabang ini tidak memiliki kolom **sub_kel**. Aplikasi menggunakan schema adapter dan analisis tetap berjalan sampai level **kel_barang**.")
    if reconstructed:
        st.info(f"Tanggal pada **{reconstructed:,} baris** direkonstruksi dari `kd_trx` karena kolom `tgl` tidak memuat tanggal kalender. Analisis per jam dinonaktifkan untuk baris tersebut.".replace(",", "."))

    summary = pd.DataFrame({"issue": list(issues.keys()), "rows": [len(v) for v in issues.values()]})
    section_title("Anomaly Summary")
    show_table(summary.sort_values("rows", ascending=False))

    cov = hpp_coverage(bundle.tx[bundle.tx["date"].le(as_of)])
    if not cov.empty:
        section_title("HPP Source Coverage")
        show_table(cov)

    issue_name = st.selectbox("Lihat detail issue", list(issues.keys()))
    detail = issues[issue_name]
    show_table(detail.head(2000))
    if not detail.empty:
        df_download_button(detail, "Download Issue Detail", f"{issue_name}.csv", f"issue_{issue_name}")


def main():
    if "bundle" not in st.session_state:
        render_upload_page()
        return
    bundle: AnalysisBundle = st.session_state["bundle"]
    page, as_of, filters = sidebar_navigation(bundle)
    location = st.session_state.get("location", "IDK-ATP")
    if page == "Command Center":
        render_command_center(bundle, as_of, location)
    elif page == "Target Chase":
        render_target_chase(bundle, as_of, location)
    elif page == "Pareto & Product Opportunity":
        render_pareto(bundle, as_of, filters)
    elif page == "Sales Performance":
        render_sales(bundle, as_of, filters)
    elif page == "Inventory & Replenishment":
        render_inventory(bundle, as_of, filters)
    elif page == "Profitability":
        render_profitability(bundle, as_of, filters)
    elif page == "Category & Supplier":
        render_category_supplier(bundle, as_of, filters)
    elif page == "SKU 360":
        render_sku360(bundle, as_of, filters)
    elif page == "Ask Anything by AI":
        render_ask_ai(bundle, as_of, location, filters)
    elif page == "AI Presentation":
        render_ai_presentation(bundle, as_of, location)
    elif page == "Data & Anomaly Center":
        render_data_anomaly(bundle, as_of)


if __name__ == "__main__":
    main()
