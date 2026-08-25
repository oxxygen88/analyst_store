from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import importlib.util
from io import BytesIO
from typing import Any, Dict

import numpy as np
import pandas as pd

from .analytics import (
    anomaly_tables,
    commercial_kpis,
    current_stock_snapshot,
    filter_period,
    inventory_health,
    monthly_sales,
    opportunity_scoring,
    pareto_products,
    revenue_inventory_matrix,
    target_status,
)
from .utils import pct, rupiah
from .gemini_models import call_gemini_with_fallback


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


def _records(df: pd.DataFrame, columns: list[str], n: int = 10) -> list[dict]:
    if df is None or df.empty:
        return []
    cols = [c for c in columns if c in df.columns]
    out = df[cols].head(n).copy()
    records = []
    for row in out.to_dict("records"):
        records.append({k: _jsonable(v) for k, v in row.items()})
    return records


def presentation_context(bundle, as_of: pd.Timestamp, location: str = "IDK-ATP") -> Dict[str, Any]:
    """Compact, aggregated context safe to send to AI. Raw transaction rows are intentionally excluded."""
    as_of = pd.Timestamp(as_of).normalize()
    start = as_of.to_period("M").to_timestamp()
    tx_period = filter_period(bundle.tx, start, as_of)
    kpi = commercial_kpis(tx_period)
    inv = inventory_health(bundle.opening, bundle.tx, as_of, bundle.purchases)
    p = opportunity_scoring(pareto_products(bundle.tx, inv, start, as_of), bundle.tx, as_of)
    target = target_status(bundle.tx, bundle.targets, start, as_of, location)
    monthly = monthly_sales(bundle.tx)
    monthly = monthly[monthly["month"].le(as_of.to_period("M").to_timestamp())].copy()
    if bundle.targets is not None and not bundle.targets.empty:
        monthly = monthly.merge(
            bundle.targets[["bulan", "target_omzet"]].rename(columns={"bulan": "month"}),
            on="month",
            how="left",
        )
    else:
        monthly["target_omzet"] = np.nan

    inv_summary = (
        inv.groupby("inventory_status", as_index=False)
        .agg(sku_count=("sku", "nunique"), stock_qty=("current_stock", "sum"), stock_value=("current_stock_value", lambda s: float(s.clip(lower=0).sum())))
        .sort_values("stock_value", ascending=False)
    )

    core = p[p["pareto_group"].eq("CORE_20")] if not p.empty else pd.DataFrame()
    opp = p[p["pareto_group"].eq("OPPORTUNITY")] if not p.empty else pd.DataFrame()
    longtail = p[p["pareto_group"].eq("LONG_TAIL")] if not p.empty else pd.DataFrame()
    total_rev = float(p["revenue"].sum()) if not p.empty else 0.0
    core_share = float(core["revenue"].sum() / total_rev) if total_rev else 0.0
    opp_share = float(opp["revenue"].sum() / total_rev) if total_rev else 0.0
    a80_count = int(p["a80_member"].sum()) if not p.empty else 0

    focus = p[p["recommended_action"].isin([
        "Push sales", "Push / campaign candidate", "Replenish before push",
        "Protect sales / replenish", "Emergency replenish",
    ])].copy() if not p.empty else pd.DataFrame()
    if not focus.empty:
        focus["priority"] = focus["recommended_action"].map({
            "Emergency replenish": 1,
            "Protect sales / replenish": 2,
            "Push sales": 3,
            "Push / campaign candidate": 4,
            "Replenish before push": 5,
        }).fillna(9)
        focus = focus.sort_values(["priority", "opportunity_score", "revenue"], ascending=[True, False, False])

    supplier = revenue_inventory_matrix(bundle.tx, inv, start, as_of, "supplier")
    category = revenue_inventory_matrix(bundle.tx, inv, start, as_of, "subdept")
    issues = anomaly_tables(bundle.opening, bundle.tx[bundle.tx["date"].le(as_of)], inv)

    target_dict = None
    if target is not None:
        target_dict = {
            "target": target.target,
            "actual": target.actual,
            "achievement": target.achievement,
            "gap": target.gap,
            "daily_run_rate": target.daily_run_rate,
            "required_daily_sales": target.required_daily_sales,
            "projected_month_end": target.projected_month_end,
            "projected_gap": target.projected_gap,
            "pace_achievement": target.pace_achievement,
            "remaining_days": target.remaining_days,
        }

    return {
        "branch": location,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "period": {"start": start.strftime("%Y-%m-%d"), "end": as_of.strftime("%Y-%m-%d")},
        "kpi": {k: _jsonable(v) for k, v in kpi.items()},
        "target": target_dict,
        "pareto": {
            "active_selling_sku": int(len(p)),
            "core20_sku": int(len(core)),
            "core_share": core_share,
            "opportunity_sku": int(len(opp)),
            "opportunity_share": opp_share,
            "long_tail_sku": int(len(longtail)),
            "a80_sku": a80_count,
            "core_stockout": int((core["current_stock"].le(0)).sum()) if not core.empty else 0,
        },
        "monthly": _records(monthly, ["month", "net_sales", "target_omzet", "trx_count", "atv", "gross_profit", "gross_margin"], 24),
        "top_focus_products": _records(focus, ["sku", "nama_barang", "supplier", "pareto_group", "revenue", "revenue_share", "growth_30d", "gross_margin", "current_stock", "stock_cover_days", "inventory_status", "opportunity_score", "recommended_action"], 15),
        "inventory_summary": _records(inv_summary, ["inventory_status", "sku_count", "stock_qty", "stock_value"], 20),
        "top_suppliers": _records(supplier, ["supplier", "revenue", "inventory_value", "revenue_share", "inventory_share", "productivity_index"], 10),
        "top_categories": _records(category, ["subdept", "revenue", "inventory_value", "revenue_share", "inventory_share", "productivity_index"], 10),
        "anomalies": {name: int(len(df)) for name, df in issues.items()},
    }


def _extract_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise ValueError("Respons AI tidak berisi JSON yang dapat dibaca.")
        return json.loads(m.group(0))


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def ai_runtime_info() -> Dict[str, bool]:
    """Return dependency status without importing heavy packages."""
    return {
        "openai_sdk": _module_available("openai"),
        "gemini_sdk": _module_available("google.genai"),
        "python_pptx": _module_available("pptx"),
    }


def _responses_output_text(payload: Dict[str, Any]) -> str:
    """Extract text from the raw JSON returned by the Responses API."""
    chunks = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    text = "\n".join(chunks).strip()
    if not text and payload.get("output_text"):
        text = str(payload["output_text"]).strip()
    if not text:
        raise RuntimeError("OpenAI mengembalikan respons tanpa teks yang dapat dibaca.")
    return text


def _call_openai_responses_http(api_key: str, model: str, prompt: str) -> str:
    """Dependency-free HTTPS fallback when the openai Python SDK is unavailable."""
    body = json.dumps({"model": model, "input": prompt}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
            raise RuntimeError(f"OpenAI API menolak request karena limit/billing (429): {message}") from exc
        raise RuntimeError(f"OpenAI API error HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tidak dapat terhubung ke OpenAI API: {exc.reason}") from exc
    return _responses_output_text(payload)



def _call_gemini_http(api_key: str, model: str, prompt: str) -> str:
    """Gemini REST call with stable-model auto fallback."""
    return call_gemini_with_fallback(
        api_key, model, prompt, json_mode=True, prefer_sdk=False
    )


def _call_gemini_sdk(api_key: str, model: str, prompt: str) -> str:
    """Gemini SDK call with stable-model auto fallback."""
    return call_gemini_with_fallback(
        api_key, model, prompt, json_mode=True, prefer_sdk=True
    )

def generate_ai_insights(api_key: str, model: str, context: Dict[str, Any], *, audience: str, language: str, provider: str = "openai") -> Dict[str, Any]:
    """Generate structured management insight using OpenAI or Google Gemini."""
    provider_key = (provider or "openai").strip().lower()
    if provider_key not in {"openai", "gemini"}:
        raise ValueError(f"AI provider tidak dikenal: {provider}")
    if not api_key:
        provider_name = "Gemini" if provider_key == "gemini" else "OpenAI"
        raise ValueError(f"{provider_name} API key belum diisi.")

    lang_instruction = "Bahasa Indonesia yang profesional dan ringkas" if language == "Bahasa Indonesia" else "professional concise English"
    schema = {
        "presentation_title": "string",
        "executive_summary": ["3-5 concise strings"],
        "target_insights": ["2-4 strings"],
        "pareto_insights": ["2-4 strings"],
        "inventory_insights": ["2-4 strings"],
        "profitability_insights": ["1-3 strings"],
        "risks": ["2-4 strings"],
        "recommended_actions": [
            {"priority": 1, "action": "string", "why": "string", "expected_impact": "string", "owner": "string"}
        ],
        "closing_message": "string",
    }
    prompt = f"""
You are a senior retail performance analyst preparing a management presentation for a Mom & Baby retail branch.
Audience: {audience}
Write in: {lang_instruction}.

STRICT DATA RULES:
- Use ONLY the numbers and facts in CONTEXT below.
- Do not invent revenue, margin, targets, product performance, or causal explanations.
- If causality is not proven, phrase it as a hypothesis/recommendation, not a fact.
- Prioritize actions that help close the monthly target while protecting Core products and avoiding unhealthy inventory.
- Treat the Pareto result dynamically: Core 20% is not automatically equal to 80% revenue.
- HPP/Gross Profit are estimates when present.
- Focus on management-ready insights, not generic advice.
- Do not mention that you are an AI.

Return ONLY valid JSON matching this shape:
{json.dumps(schema, ensure_ascii=False)}

CONTEXT:
{json.dumps(context, ensure_ascii=False, default=_jsonable)}
""".strip()

    if provider_key == "gemini":
        if _module_available("google.genai"):
            response_text = _call_gemini_sdk(api_key, model, prompt)
        else:
            response_text = _call_gemini_http(api_key, model, prompt)
    else:
        # Prefer the official SDK, but do not make the entire AI menu depend on it.
        try:
            from openai import OpenAI
        except ImportError:
            response_text = _call_openai_responses_http(api_key, model, prompt)
        else:
            try:
                client = OpenAI(api_key=api_key, timeout=120.0)
                response = client.responses.create(model=model, input=prompt)
                response_text = response.output_text
            except Exception as exc:
                raise RuntimeError(f"OpenAI API request gagal: {exc}") from exc

    data = _extract_json(response_text)

    # Defensive normalization so the PPT builder always receives predictable structures.
    for key in ["executive_summary", "target_insights", "pareto_insights", "inventory_insights", "profitability_insights", "risks"]:
        val = data.get(key, [])
        if not isinstance(val, list):
            val = [str(val)]
        data[key] = [str(x) for x in val if str(x).strip()][:6]
    actions = data.get("recommended_actions", [])
    if not isinstance(actions, list):
        actions = []
    norm_actions = []
    for i, row in enumerate(actions[:8], start=1):
        if isinstance(row, dict):
            norm_actions.append({
                "priority": row.get("priority", i),
                "action": str(row.get("action", "")),
                "why": str(row.get("why", "")),
                "expected_impact": str(row.get("expected_impact", "")),
                "owner": str(row.get("owner", "Management")),
            })
    data["recommended_actions"] = norm_actions
    return data


# ---------- PowerPoint builder ----------

def build_pptx(context: Dict[str, Any], ai: Dict[str, Any], *, branch: str, as_of: pd.Timestamp, audience: str) -> bytes:
    try:
        from pptx import Presentation
        from pptx.chart.data import CategoryChartData
        from pptx.dml.color import RGBColor
        from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
        from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
        from pptx.util import Inches, Pt
    except ImportError as exc:
        raise RuntimeError("Package 'python-pptx' belum terinstall. Jalankan setup_windows.bat kembali setelah update aplikasi.") from exc

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    BLUE = RGBColor(47, 104, 176)
    ORANGE = RGBColor(246, 139, 36)
    DARK = RGBColor(31, 41, 55)
    MUTED = RGBColor(100, 116, 139)
    LIGHT = RGBColor(244, 247, 251)
    WHITE = RGBColor(255, 255, 255)
    GREEN = RGBColor(22, 163, 74)
    RED = RGBColor(220, 38, 38)

    def add_rect(slide, x, y, w, h, fill, radius=False):
        from pptx.enum.shapes import MSO_SHAPE
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = fill
        shape.line.fill.background()
        return shape

    def add_text(slide, text, x, y, w, h, size=18, color=DARK, bold=False, align=None, valign=None):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame; tf.clear(); tf.word_wrap = True
        if valign is not None: tf.vertical_anchor = valign
        p = tf.paragraphs[0]; p.text = str(text)
        p.font.name = "Aptos"; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
        if align is not None: p.alignment = align
        return box

    def slide_header(slide, title, subtitle=None, num=None):
        add_text(slide, title, .7, .35, 11.7, .5, 25, DARK, True)
        if subtitle:
            add_text(slide, subtitle, .72, .9, 11.7, .32, 10.5, MUTED)
        add_rect(slide, .7, 1.26, 1.0, .055, ORANGE)
        if num is not None:
            add_text(slide, str(num), 12.15, .38, .45, .3, 9, MUTED, False, PP_ALIGN.RIGHT)

    def footer(slide):
        add_text(slide, f"INDOKIDS · {branch} · Data s.d. {pd.Timestamp(as_of):%d %b %Y}", .7, 7.12, 11.8, .22, 8.5, MUTED)

    def bullet_block(slide, items, x, y, w, h, size=16):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame; tf.clear(); tf.word_wrap = True
        for i, item in enumerate(items or []):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = str(item); p.level = 0
            p.font.name = "Aptos"; p.font.size = Pt(size); p.font.color.rgb = DARK
            p.space_after = Pt(9)
            p.text = "• " + p.text
        return box

    def metric_card(slide, x, y, w, title, value, note=""):
        add_rect(slide, x, y, w, 1.08, LIGHT, True)
        add_text(slide, title, x+.16, y+.13, w-.32, .22, 9.5, MUTED, True)
        add_text(slide, value, x+.16, y+.42, w-.32, .32, 19, DARK, True)
        if note:
            add_text(slide, note, x+.16, y+.79, w-.32, .18, 8.5, MUTED)

    # 1 Title
    slide = prs.slides.add_slide(blank)
    add_rect(slide, 0, 0, 13.333, 7.5, BLUE)
    add_rect(slide, .72, 1.45, .12, 2.25, ORANGE)
    add_text(slide, ai.get("presentation_title") or "Branch Performance & Target Recovery", 1.1, 1.45, 10.8, 1.3, 32, WHITE, True)
    add_text(slide, f"{branch} · Management Brief", 1.1, 2.9, 9.6, .5, 19, WHITE)
    add_text(slide, f"Data sampai {pd.Timestamp(as_of):%d %B %Y} · Audience: {audience}", 1.1, 3.52, 9.8, .35, 11, WHITE)
    add_text(slide, "Monitor · Diagnose · Act", 1.1, 5.85, 4.8, .4, 13, WHITE, True)

    # 2 Executive overview
    slide = prs.slides.add_slide(blank); slide_header(slide, "Executive Overview", "Current performance, target pace, and commercial health", 2)
    k = context["kpi"]; t = context.get("target") or {}
    metric_card(slide, .7, 1.55, 2.85, "NET SALES MTD", rupiah(k.get("net_sales")))
    metric_card(slide, 3.75, 1.55, 2.85, "TARGET", rupiah(t.get("target")) if t else "-")
    metric_card(slide, 6.8, 1.55, 2.85, "ACHIEVEMENT", pct(t.get("achievement")) if t else "-")
    metric_card(slide, 9.85, 1.55, 2.78, "PROJECTED GAP", rupiah(t.get("projected_gap")) if t else "-")
    metric_card(slide, .7, 2.85, 2.85, "TRANSACTIONS", f"{int(k.get('trx_count',0)):,}".replace(",", "."))
    metric_card(slide, 3.75, 2.85, 2.85, "AVG TRANSACTION", rupiah(k.get("atv")))
    metric_card(slide, 6.8, 2.85, 2.85, "EST. GROSS PROFIT", rupiah(k.get("gross_profit")))
    metric_card(slide, 9.85, 2.85, 2.78, "GROSS MARGIN", pct(k.get("gross_margin")))
    add_text(slide, "AI readout", .72, 4.28, 2.0, .32, 13, BLUE, True)
    bullet_block(slide, ai.get("executive_summary", []), .8, 4.72, 11.7, 1.95, 15)
    footer(slide)

    # 3 Target trend
    slide = prs.slides.add_slide(blank); slide_header(slide, "Target Performance", "Actual sales vs monthly target", 3)
    monthly = context.get("monthly", [])
    if monthly:
        chart_data = CategoryChartData()
        chart_data.categories = [pd.Timestamp(r["month"]).strftime("%b") for r in monthly]
        chart_data.add_series("Net Sales", [float(r.get("net_sales") or 0) for r in monthly])
        chart_data.add_series("Target", [float(r.get("target_omzet") or 0) for r in monthly])
        chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.75), Inches(1.6), Inches(7.55), Inches(4.85), chart_data).chart
        chart.has_legend = True; chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.value_axis.tick_labels.number_format = '0,,"M"'
        chart.value_axis.has_major_gridlines = True
        chart.chart_title.text_frame.text = ""
    add_text(slide, "What matters now", 8.7, 1.7, 3.6, .35, 14, BLUE, True)
    bullet_block(slide, ai.get("target_insights", []), 8.7, 2.2, 3.8, 3.8, 14)
    if t:
        add_text(slide, f"Required pace: {rupiah(t.get('required_daily_sales'))}/hari", 8.7, 6.1, 3.8, .3, 11, RED if (t.get("projected_gap") or 0) > 0 else GREEN, True)
    footer(slide)

    # 4 Pareto
    slide = prs.slides.add_slide(blank); slide_header(slide, "Pareto & Revenue Concentration", "Protect winners, then grow the opportunity layer", 4)
    par = context["pareto"]
    chart_data = CategoryChartData(); chart_data.categories = ["Core 20", "Opportunity", "Long Tail"]
    chart_data.add_series("Revenue Share", [par.get("core_share",0)*100, par.get("opportunity_share",0)*100, max(0,1-par.get("core_share",0)-par.get("opportunity_share",0))*100])
    chart = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(.8), Inches(1.7), Inches(5.4), Inches(3.9), chart_data).chart
    chart.has_legend = False; chart.value_axis.tick_labels.number_format = '0"%"'; chart.value_axis.maximum_scale = 100
    metric_card(slide, .85, 5.8, 2.4, "CORE 20 SHARE", pct(par.get("core_share")))
    metric_card(slide, 3.45, 5.8, 2.4, "A80 SKU", f"{par.get('a80_sku',0):,}".replace(",", "."))
    add_text(slide, "AI insights", 6.65, 1.75, 2.2, .35, 14, BLUE, True)
    bullet_block(slide, ai.get("pareto_insights", []), 6.65, 2.2, 5.8, 3.8, 14)
    footer(slide)

    # 5 Product focus table
    slide = prs.slides.add_slide(blank); slide_header(slide, "Products to Improve", "Highest-priority growth and availability actions", 5)
    rows = context.get("top_focus_products", [])[:8]
    headers = ["SKU", "Product", "Revenue", "Growth", "Stock", "Status", "Action"]
    table = slide.shapes.add_table(len(rows)+1, len(headers), Inches(.6), Inches(1.55), Inches(12.15), Inches(4.9)).table
    widths = [1.25, 3.1, 1.45, 1.0, .75, 1.25, 3.35]
    for i,w in enumerate(widths): table.columns[i].width = Inches(w)
    for c,h in enumerate(headers):
        cell=table.cell(0,c); cell.text=h; cell.fill.solid(); cell.fill.fore_color.rgb=BLUE
        for p in cell.text_frame.paragraphs: p.font.color.rgb=WHITE; p.font.bold=True; p.font.size=Pt(10)
    for r_idx,row in enumerate(rows,1):
        vals=[row.get("sku",""), str(row.get("nama_barang","") or "")[:34], rupiah(row.get("revenue")), pct(row.get("growth_30d")), f"{float(row.get('current_stock') or 0):,.0f}".replace(",","."), row.get("inventory_status",""), row.get("recommended_action","")]
        for c,val in enumerate(vals):
            cell=table.cell(r_idx,c); cell.text=str(val); cell.fill.solid(); cell.fill.fore_color.rgb=WHITE if r_idx%2 else LIGHT
            for p in cell.text_frame.paragraphs: p.font.size=Pt(8.5); p.font.color.rgb=DARK
    footer(slide)

    # 6 Inventory
    slide = prs.slides.add_slide(blank); slide_header(slide, "Inventory Health", "Revenue protection requires healthy availability and controlled capital", 6)
    invsum = context.get("inventory_summary", [])
    if invsum:
        chart_data = CategoryChartData(); chart_data.categories=[str(r.get("inventory_status","")) for r in invsum]
        chart_data.add_series("SKU", [int(r.get("sku_count") or 0) for r in invsum])
        chart=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.75), Inches(1.65), Inches(7.0), Inches(4.65), chart_data).chart
        chart.has_legend=False; chart.value_axis.has_major_gridlines=True
    add_text(slide, "AI inventory readout", 8.15, 1.72, 3.4, .35, 14, BLUE, True)
    bullet_block(slide, ai.get("inventory_insights", []), 8.15, 2.18, 4.3, 2.55, 14)
    add_text(slide, "Risks", 8.15, 4.9, 2.0, .3, 13, RED, True)
    bullet_block(slide, ai.get("risks", []), 8.15, 5.25, 4.3, 1.25, 12.5)
    footer(slide)

    # 7 Action plan
    slide = prs.slides.add_slide(blank); slide_header(slide, "Recommended Actions", "Prioritized actions generated from the current branch data", 7)
    actions = ai.get("recommended_actions", [])[:6]
    y=1.55
    for i,a in enumerate(actions,1):
        add_rect(slide,.75,y,11.85,.72,LIGHT,True)
        add_text(slide,str(a.get("priority",i)),.95,y+.16,.35,.26,12,BLUE,True,PP_ALIGN.CENTER)
        add_text(slide,a.get("action",""),1.4,y+.10,4.25,.25,12,DARK,True)
        add_text(slide,a.get("why",""),1.4,y+.38,5.8,.2,8.5,MUTED)
        add_text(slide,a.get("expected_impact",""),7.35,y+.14,3.2,.2,9.5,DARK,True)
        add_text(slide,a.get("owner",""),10.75,y+.14,1.4,.2,9,MUTED,True)
        y += .83
    footer(slide)

    # 8 Closing
    slide = prs.slides.add_slide(blank)
    add_rect(slide,0,0,13.333,7.5,DARK)
    add_text(slide,"Management Focus",.85,1.05,5.5,.6,29,WHITE,True)
    add_rect(slide,.87,1.85,1.0,.06,ORANGE)
    add_text(slide,ai.get("closing_message") or "Protect winners. Grow potentials. Fix availability. Reduce waste.",.87,2.4,10.8,1.4,25,WHITE,True)
    add_text(slide,"Next review: validate completed actions against revenue recovered and target pace.",.9,5.65,10.5,.4,12,WHITE)

    bio=BytesIO(); prs.save(bio); return bio.getvalue()
