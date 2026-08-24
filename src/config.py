APP_TITLE = "INDOKIDS Branch Performance Command Center"
APP_SUBTITLE = "Monitor · Diagnose · Act"

REQUIRED_OPENING_COLUMNS = {
    "sku", "nama_barang", "supplier", "subdept", "kel_barang", "sub_kel",
    "saldo_awal", "hrg_beli", "subtotal",
}

REQUIRED_TRANSACTION_COLUMNS = {
    "kd_trx", "tgl", "sku", "nama_barang", "supplier", "subdept", "kel_barang",
    "sub_kel", "stock_in", "stock_out", "harga", "subtotal", "keterangan",
}

REQUIRED_PURCHASE_COLUMNS = {"tgl", "no_faktur_beli", "sku", "harga_beli"}
REQUIRED_TARGET_COLUMNS = {"bulan", "target_omzet"}

MOVEMENT_LABELS = {
    "SALE": "Penjualan",
    "SALES_RETURN": "Return Penjualan",
    "PURCHASE": "Pembelian",
    "PRE_RECEIVE": "Pembelian Pre Receive",
    "TRANSFER_IN": "Mutasi Masuk",
    "TRANSFER_OUT": "Mutasi Keluar",
    "ADJUSTMENT_IN": "Adjustment Masuk",
    "ADJUSTMENT_OUT": "Adjustment Keluar",
    "ADJUSTMENT_MATCH": "Adjustment Tanpa Perubahan",
    "OPNAME_IN": "Opname Masuk",
    "OPNAME_OUT": "Opname Keluar",
    "OPNAME_MATCH": "Opname Sesuai",
    "OTHER_IN": "Lainnya Masuk",
    "OTHER_OUT": "Lainnya Keluar",
    "OTHER_MATCH": "Lainnya Tanpa Perubahan",
}

PARETO_CORE_SHARE = 0.20
PARETO_REVENUE_SHARE = 0.80

# Opportunity Score weights. Margin weight is redistributed automatically when HPP unavailable.
OPPORTUNITY_WEIGHTS = {
    "revenue": 0.30,
    "growth": 0.20,
    "margin": 0.20,
    "stock": 0.20,
    "consistency": 0.10,
}

# Heuristics only for action prioritisation; inventory status itself follows company sigma rules.
LOW_COVER_DAYS = 14
HEALTHY_COVER_DAYS = 60
HIGH_COVER_DAYS = 90

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
