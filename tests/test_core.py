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
                "AIza-test", "gemini-2.5-flash", {"kpi": {"net_sales": 1}},
                audience="Management / Owner", language="Bahasa Indonesia", provider="gemini"
            )
    assert result["presentation_title"] == "Gemini Test"
    assert http_call.called
