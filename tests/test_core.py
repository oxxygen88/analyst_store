import pandas as pd
import numpy as np

from src.transform import enrich_transactions
from src.analytics import inventory_health, pareto_products, opportunity_scoring, target_status
from src.hpp import resolve_commercial_hpp


def synthetic_data():
    opening = pd.DataFrame({
        "sku":["A","B","C"],
        "nama_barang":["Alpha","Beta","Gamma"],
        "supplier":["S1","S1","S2"],
        "subdept":["D1","D1","D2"],
        "kel_barang":["K1","K1","K2"],
        "sub_kel":["SK1","SK1","SK2"],
        "saldo_awal":[10.0,5.0,0.0],
        "hrg_beli":[50.0,30.0,0.0],
        "subtotal":[500.0,150.0,0.0],
    })
    rows=[]
    def add(kd,tgl,sku,si,so,harga,ket,name,sup,dept,kel,sub):
        rows.append([kd,pd.Timestamp(tgl),sku,name,sup,dept,kel,sub,si,so,harga,(si or so)*harga,ket])
    add("J1","2026-01-05","A",0,2,100,"Penjualan","Alpha","S1","D1","K1","SK1")
    add("B1","2026-01-10","A",5,0,55,"Pembelian","Alpha","S1","D1","K1","SK1")
    add("J2","2026-02-02","A",0,3,110,"Penjualan","Alpha","S1","D1","K1","SK1")
    add("J3","2026-02-03","B",0,1,70,"Penjualan","Beta","S1","D1","K1","SK1")
    add("M1","2026-02-05","C",8,0,20,"Mutasi Dari HO","Gamma","S2","D2","K2","SK2")
    add("J4","2026-02-06","C",0,2,50,"Penjualan","Gamma","S2","D2","K2","SK2")
    tx=pd.DataFrame(rows,columns=["kd_trx","tgl","sku","nama_barang","supplier","subdept","kel_barang","sub_kel","stock_in","stock_out","harga","subtotal","keterangan"])
    purchases=pd.DataFrame({"tgl":[pd.Timestamp("2026-01-10")],"no_faktur_beli":["B1"],"sku":["A"],"harga_beli":[55.0]})
    targets=pd.DataFrame({"bulan":[pd.Timestamp("2026-02-01")],"lokasi":["IDK-ATP"],"target_omzet":[1000.0]})
    return opening,tx,purchases,targets


def test_movement_and_hpp():
    opening,tx,purchases,_=synthetic_data()
    tx=enrich_transactions(tx)
    tx=resolve_commercial_hpp(tx,opening,purchases)
    assert tx.loc[tx.kd_trx.eq("J1"),"movement"].iloc[0] == "SALE"
    # J1 predates purchase, so opening cost is used.
    assert tx.loc[tx.kd_trx.eq("J1"),"hpp_unit"].iloc[0] == 50
    # J2 uses last purchase.
    assert tx.loc[tx.kd_trx.eq("J2"),"hpp_unit"].iloc[0] == 55
    # C uses transfer-in cost.
    assert tx.loc[tx.kd_trx.eq("J4"),"hpp_unit"].iloc[0] == 20


def test_inventory_and_pareto():
    opening,tx,purchases,_=synthetic_data()
    tx=resolve_commercial_hpp(enrich_transactions(tx),opening,purchases)
    inv=inventory_health(opening,tx,pd.Timestamp("2026-02-28"),purchases)
    a=inv.set_index("sku").loc["A"]
    assert a.current_stock == 10
    p=opportunity_scoring(pareto_products(tx,inv,pd.Timestamp("2026-02-01"),pd.Timestamp("2026-02-28")),tx,pd.Timestamp("2026-02-28"))
    assert not p.empty
    assert set(p.pareto_group).issubset({"CORE_20","OPPORTUNITY","LONG_TAIL"})


def test_target_status():
    opening,tx,purchases,targets=synthetic_data()
    tx=resolve_commercial_hpp(enrich_transactions(tx),opening,purchases)
    status=target_status(tx,targets,pd.Timestamp("2026-02-01"),pd.Timestamp("2026-02-28"),"IDK-ATP")
    assert status is not None
    assert status.target == 1000
    assert status.actual > 0


def test_rupiah_format():
    from src.utils import rupiah
    assert rupiah(1234567) == "Rp. 1.234.567"
    assert rupiah(-1234567) == "-Rp. 1.234.567"


def test_pptx_builder_mock():
    from src.ai_presentation import build_pptx
    context = {
        "kpi": {"net_sales": 1000000.0, "trx_count": 10, "atv": 100000.0, "gross_profit": 250000.0, "gross_margin": .25},
        "target": {"target": 1500000.0, "achievement": 2/3, "projected_gap": 200000.0, "required_daily_sales": 50000.0},
        "pareto": {"core_share": .60, "opportunity_share": .20, "a80_sku": 20},
        "monthly": [{"month":"2026-01-01","net_sales":1000000.0,"target_omzet":1500000.0}],
        "top_focus_products": [{"sku":"A","nama_barang":"Alpha","revenue":500000.0,"growth_30d":.2,"current_stock":5,"inventory_status":"NORMAL","recommended_action":"Push sales"}],
        "inventory_summary": [{"inventory_status":"NORMAL","sku_count":100,"stock_qty":500,"stock_value":2000000.0}],
    }
    ai = {
        "presentation_title":"Test Deck",
        "executive_summary":["Test summary"],
        "target_insights":["Test target"],
        "pareto_insights":["Test pareto"],
        "inventory_insights":["Test inventory"],
        "risks":["Test risk"],
        "recommended_actions":[{"priority":1,"action":"Push Alpha","why":"Potential","expected_impact":"Growth","owner":"Buyer"}],
        "closing_message":"Close",
    }
    payload = build_pptx(context, ai, branch="IDK-ATP", as_of=pd.Timestamp("2026-08-22"), audience="Management")
    assert payload[:2] == b"PK"
    assert len(payload) > 10000


def test_ai_sdk_missing_uses_http_fallback():
    import sys
    from unittest.mock import patch
    from src.ai_presentation import generate_ai_insights

    raw = '{"presentation_title":"Test","executive_summary":["OK"],"target_insights":[],"pareto_insights":[],"inventory_insights":[],"profitability_insights":[],"risks":[],"recommended_actions":[],"closing_message":"Done"}'
    with patch.dict(sys.modules, {"openai": None}):
        with patch("src.ai_presentation._call_openai_responses_http", return_value=raw) as http_call:
            result = generate_ai_insights(
                "sk-test", "gpt-5.6", {"kpi": {"net_sales": 1}},
                audience="Management / Owner", language="Bahasa Indonesia"
            )
    assert result["presentation_title"] == "Test"
    assert http_call.called


def test_gemini_sdk_missing_uses_http_fallback():
    from unittest.mock import patch
    from src.ai_presentation import generate_ai_insights

    raw = '{"presentation_title":"Gemini Test","executive_summary":["OK"],"target_insights":[],"pareto_insights":[],"inventory_insights":[],"profitability_insights":[],"risks":[],"recommended_actions":[],"closing_message":"Done"}'
    with patch("src.ai_presentation._module_available", return_value=False):
        with patch("src.ai_presentation._call_gemini_http", return_value=raw) as http_call:
            result = generate_ai_insights(
                "AIza-test", "gemini-3.7-flash", {"kpi": {"net_sales": 1}},
                audience="Management / Owner", language="Bahasa Indonesia", provider="gemini"
            )
    assert result["presentation_title"] == "Gemini Test"
    assert http_call.called


def test_ai_analyst_context_supplier_month():
    from src.ai_analyst import build_question_context
    from src.pipeline import AnalysisBundle
    from src.transform import build_master, daily_movement

    opening, tx, purchases, targets = synthetic_data()
    tx = resolve_commercial_hpp(enrich_transactions(tx), opening, purchases)
    bundle = AnalysisBundle(
        opening=opening,
        tx=tx,
        purchases=purchases,
        targets=targets,
        master=build_master(opening, tx),
        daily=daily_movement(tx),
        min_date=tx["date"].min(),
        max_date=tx["date"].max(),
    )
    ctx, tables, scope = build_question_context(
        bundle,
        "Berapa penjualan supplier S1 bulan Februari 2026?",
        pd.Timestamp("2026-02-28"),
        location="IDK-ATP",
    )
    assert scope["period"]["start"] == "2026-02-01"
    assert scope["period"]["end"] == "2026-02-28"
    assert ctx["applied_filters"]["supplier"] == ["S1"]
    assert round(ctx["commercial_kpi"]["net_sales"], 2) == 400.0
    assert "product_sales" in tables


def test_ai_analyst_inventory_status_filter():
    from src.ai_analyst import build_question_context
    from src.pipeline import AnalysisBundle
    from src.transform import build_master, daily_movement

    opening, tx, purchases, targets = synthetic_data()
    tx = resolve_commercial_hpp(enrich_transactions(tx), opening, purchases)
    bundle = AnalysisBundle(
        opening=opening,
        tx=tx,
        purchases=purchases,
        targets=targets,
        master=build_master(opening, tx),
        daily=daily_movement(tx),
        min_date=tx["date"].min(),
        max_date=tx["date"].max(),
    )
    ctx, tables, scope = build_question_context(
        bundle,
        "Tampilkan item slow moving",
        pd.Timestamp("2026-02-28"),
        location="IDK-ATP",
    )
    assert "SLOW" in scope["status_filters"]
    assert "inventory_detail" in tables
    if not tables["inventory_detail"].empty:
        assert set(tables["inventory_detail"]["inventory_status"].unique()) <= {"SLOW"}


def test_ai_analyst_excel_export():
    from src.ai_analyst import tables_to_excel
    payload = tables_to_excel({"sales": pd.DataFrame({"sku": ["A"], "net_sales": [1000]})})
    assert payload[:2] == b"PK"
    assert len(payload) > 1000


def test_ai_analyst_openai_fallback_mock():
    from unittest.mock import patch
    from src.ai_analyst import generate_analyst_answer
    with patch("src.ai_analyst._call_openai_text", return_value="Jawaban berdasarkan data") as call:
        out = generate_analyst_answer(
            "sk-test", "gpt-5.6", "openai", "Berapa omzet?",
            {"commercial_kpi": {"net_sales": 1000}}, history=[]
        )
    assert out == "Jawaban berdasarkan data"
    assert call.called


def test_gemini_model_fallback_on_unavailable():
    from unittest.mock import patch
    from src.gemini_models import call_gemini_with_fallback, GeminiModelUnavailableError

    with patch("src.gemini_models.module_available", return_value=True):
        with patch("src.gemini_models._sdk_single") as sdk_call:
            sdk_call.side_effect = [
                GeminiModelUnavailableError("404 NOT_FOUND model gemini-3.7-flash unavailable"),
                "fallback success",
            ]
            out = call_gemini_with_fallback(
                "AIza-test", "gemini-3.7-flash", "hello", prefer_sdk=True
            )
    assert out == "fallback success"
    assert sdk_call.call_count == 2
    assert sdk_call.call_args_list[0].args[1] == "gemini-3.7-flash"
    assert sdk_call.call_args_list[1].args[1] == "gemini-3.6-flash"


def test_gemini_does_not_fallback_on_quota_error():
    from unittest.mock import patch
    from src.gemini_models import call_gemini_with_fallback

    with patch("src.gemini_models.module_available", return_value=True):
        with patch("src.gemini_models._sdk_single", side_effect=RuntimeError("429 quota exceeded")) as sdk_call:
            try:
                call_gemini_with_fallback(
                    "AIza-test", "gemini-3.7-flash", "hello", prefer_sdk=True
                )
                assert False, "quota error should propagate"
            except RuntimeError as exc:
                assert "429" in str(exc)
    assert sdk_call.call_count == 1


def test_gemini_fallback_chain_never_escalates_lite():
    from src.gemini_models import fallback_chain
    assert fallback_chain("gemini-3.7-flash") == [
        "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"
    ]
    assert fallback_chain("gemini-3.5-flash-lite") == ["gemini-3.5-flash-lite"]


def test_optional_sub_kel_and_transaction_date_fallback():
    from src.io import load_opening_bytes, load_transactions_bytes
    opening_csv = b"sku,nama_barang,supplier,subdept,kel_barang,saldo_awal,hrg_beli,subtotal\nA,Alpha,S1,D1,K1,10,50,500\n"
    tx_csv = b"kd_trx,tgl,sku,nama_barang,supplier,subdept,kel_barang,stock_in,stock_out,harga,subtotal,keterangan\nTHJ2601010001C,16:44.0,A,Alpha,S1,D1,K1,0,1,100,100,Penjualan\n"
    op = load_opening_bytes(opening_csv)
    tx = load_transactions_bytes(tx_csv, analysis_year=2026)
    assert "sub_kel" in op.columns and op.loc[0, "sub_kel"] == "(Tidak tersedia)"
    assert "sub_kel" in tx.columns and tx.loc[0, "sub_kel"] == "(Tidak tersedia)"
    assert tx.loc[0, "tgl"] == pd.Timestamp("2026-01-01")
    assert tx.loc[0, "date_parse_status"] == "KD_TRX_DATE"
    assert bool(tx.loc[0, "time_available"]) is False


def test_full_datetime_keeps_hour():
    from src.io import load_transactions_bytes
    tx_csv = b"kd_trx,tgl,sku,nama_barang,supplier,subdept,kel_barang,stock_in,stock_out,harga,subtotal,keterangan\nJA-2601010001-H,2026-01-01 08:39:08,A,Alpha,S1,D1,K1,0,1,100,100,Penjualan\n"
    tx = load_transactions_bytes(tx_csv, analysis_year=2026)
    assert tx.loc[0, "tgl"] == pd.Timestamp("2026-01-01 08:39:08")
    assert tx.loc[0, "date_parse_status"] == "TGL"
    assert bool(tx.loc[0, "time_available"]) is True


def test_outside_year_reference_is_flagged_not_double_counted():
    from src.io import load_transactions_bytes
    tx_csv = b"kd_trx,tgl,sku,nama_barang,supplier,subdept,kel_barang,stock_in,stock_out,harga,subtotal,keterangan\nBL-2410290028-HJ,00:00.0,A,Alpha,S1,D1,K1,1,0,50,50,Pembelian\n"
    tx = load_transactions_bytes(tx_csv, analysis_year=2026)
    assert pd.isna(tx.loc[0, "tgl"])
    assert tx.loc[0, "date_parse_status"] == "OUTSIDE_ANALYSIS_YEAR"


def test_v28_recommended_slide_plan_depth_and_mandatory():
    from src.ai_presentation import build_recommended_slide_plan
    plan = build_recommended_slide_plan(
        ["Executive Performance", "Target & Forecast", "Sales Growth", "Pareto Product", "Product Opportunity", "Supplier Performance", "30-Day Action Plan"],
        "Executive — 7–9 slides",
        "Management / Owner",
    )
    assert len(plan) <= 9
    types = [x["slide_type"] for x in plan]
    assert types[0] == "cover"
    assert "executive" in types
    assert "action_plan" in types
    assert types[-1] == "closing"


def test_v28_normalize_slide_plan_filters_invalid_and_order():
    from src.ai_presentation import normalize_slide_plan
    raw = [
        {"include": True, "order": 3, "slide_type": "target", "title": "T", "objective": "O"},
        {"include": False, "order": 1, "slide_type": "supplier", "title": "S"},
        {"include": True, "order": 2, "slide_type": "not_valid", "title": "Bad"},
        {"include": True, "order": 1, "slide_type": "cover", "title": "Cover"},
    ]
    out = normalize_slide_plan(raw)
    assert [x["slide_type"] for x in out] == ["cover", "target"]
    assert [x["order"] for x in out] == [1, 2]


def test_v28_dynamic_pptx_builder_mock():
    from src.ai_presentation import build_dynamic_pptx, build_recommended_slide_plan
    context = {
        "kpi": {"net_sales": 1000000.0, "trx_count": 10, "atv": 100000.0, "upt": 2.0, "gross_profit": 250000.0, "gross_margin": .25},
        "target": {"target": 1500000.0, "actual":1000000.0, "achievement": 2/3, "gap":500000.0, "projected_month_end":1300000.0, "projected_gap": 200000.0, "required_daily_sales": 50000.0},
        "pareto": {"core_share": .60, "opportunity_share": .20, "a80_sku": 20},
        "monthly": [{"month":"2026-01-01","net_sales":1000000.0,"target_omzet":1500000.0}],
        "top_focus_products": [{"sku":"A","nama_barang":"Alpha","pareto_group":"OPPORTUNITY","revenue":500000.0,"growth_30d":.2,"current_stock":5,"inventory_status":"NORMAL","recommended_action":"Push sales"}],
        "stockout_recovery": [],
        "inventory_summary": [{"inventory_status":"NORMAL","sku_count":100,"stock_qty":500,"stock_value":2000000.0}],
        "inventory_capital": [{"inventory_status":"NORMAL","sku_count":100,"stock_qty":500,"stock_value":2000000.0}],
        "commercial_comparison": {"metrics": {"net_sales":{"delta":.1},"trx_count":{"delta":.05},"atv":{"delta":.03},"upt":{"delta":.02}}},
        "profit_products": [{"sku":"A","nama_barang":"Alpha","revenue":500000.0,"gross_profit":150000.0,"gross_margin":.3}],
        "top_suppliers": [{"supplier":"S1","revenue":700000.0,"inventory_value":800000.0,"revenue_share":.7,"inventory_share":.4,"productivity_index":1.75}],
        "top_categories": [{"subdept":"D1","revenue":700000.0,"inventory_value":800000.0,"revenue_share":.7,"inventory_share":.4,"productivity_index":1.75}],
        "purchase_by_supplier": [], "transfer_by_supplier": [], "anomalies": {},
        "pareto_migration": {"top_changes": []},
    }
    plan = build_recommended_slide_plan(
        ["Executive Performance","Target & Forecast","Pareto Product","Product Opportunity","Inventory Health","30-Day Action Plan"],
        "Executive — 7–9 slides", "Management / Owner"
    )
    ai = {
        "presentation_title":"Dynamic Test",
        "slides":[{"slide_type":x["slide_type"],"title":x["title"],"headline":"Headline","bullets":["Point 1","Point 2"]} for x in plan],
        "recommended_actions":[{"priority":1,"action":"Push Alpha","why":"Potential","expected_impact":"Growth","owner":"Buyer","timing":"7 days"}],
        "closing_message":"Close",
    }
    payload = build_dynamic_pptx(context, ai, plan, branch="IDK-ATP", as_of=pd.Timestamp("2026-08-22"), audience="Management / Owner")
    assert payload[:2] == b"PK"
    assert len(payload) > 10000
