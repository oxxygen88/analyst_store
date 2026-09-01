# INDOKIDS Branch Performance Command Center — V2.6

Aplikasi analisis cabang retail untuk **Monitor → Diagnose → Act**, dengan fokus target achievement, Pareto product opportunity, inventory health, profitability, dan action list.

## Fitur utama

1. **Command Center** — target, actual MTD, achievement, pace, projection, required daily sales, management alerts.
2. **Target Chase** — projected gap, revenue recovery simulator, dan action list SKU.
3. **Pareto & Product Opportunity** — Dynamic Pareto, Core 20%, A80 aktual, Opportunity Products, Opportunity Score 0–100.
4. **Sales Performance** — revenue, transaksi, ATV, UPT, daily/weekday/hourly pattern.
5. **Inventory & Replenishment** — current stock, stock value, stockout, slow/dead/overstock berbasis standar deviasi perusahaan.
6. **Profitability** — Estimated HPP, gross profit, gross margin, dan HPP source coverage.
7. **Category & Supplier** — revenue share vs inventory share / productivity index.
8. **SKU 360** — histori movement, revenue, profit, stock balance, cover, dan **filter inventory status**.
9. **Ask Anything by AI** — chat Business Analyst berbasis data aplikasi dengan provider **Google Gemini / OpenAI**, fact-pack lokal, supporting tables, dan download Excel.
10. **AI Presentation Generator** — pilih **Google Gemini atau OpenAI** untuk membuat insight + PowerPoint `.pptx` management brief berbasis data aplikasi.
11. **Data & Anomaly Center** — negative stock, zero price, mismatch subtotal, duplicate suspicion, missing HPP, unknown movement.





## Update V2.6 — Gemini 3.x Compatibility & Auto Fallback

- Mengganti pilihan model Gemini lama 2.5 dengan model stabil Gemini 3.x.
- Default baru: `gemini-3.7-flash`.
- Fallback berjenjang: `3.7 Flash → 3.6 Flash → 3.5 Flash → 3.5 Flash-Lite`.
- Fallback hanya terjadi ketika API menyatakan model tidak tersedia/deprecated/404 NOT_FOUND.
- Error API key, quota 429, malformed request, atau jaringan tetap ditampilkan agar masalah asli tidak tertutupi.
- Menghapus sampling parameter deprecated untuk Gemini 3.x agar kompatibel dengan API terbaru.
- Session state lama yang masih menyimpan model 2.5 otomatis di-reset ke `gemini-3.7-flash`.
- Berlaku untuk **Ask Anything by AI** dan **AI Presentation Generator**.

## Update V2.4 — Ask Anything by AI

- Tambah menu **Ask Anything by AI** sebagai Business Analyst chat berbasis data cabang yang sudah diproses aplikasi.
- Provider AI: **Google Gemini** atau **OpenAI**, sama seperti menu AI Presentation.
- Aplikasi membangun **fact pack lokal** berdasarkan pertanyaan: periode, supplier/category/SKU yang disebut, sales, target, Pareto, inventory, movement, profitability, dan anomaly.
- Raw transaction tidak dikirim langsung ke provider AI. Hanya hasil aggregasi/tabel relevan yang dikirim.
- Mendukung pertanyaan follow-up dengan chat history dan pewarisan periode/scope untuk referensi seperti “supplier itu” atau “bulan tersebut”.
- Mendukung filter status inventory seperti `OVERSTOCK`, `DEAD`, `SLOW`, `STOCKOUT`, `NEGATIVE`, `NO_SALES`.
- Data pendukung jawaban dapat diperiksa langsung dan **di-download sebagai Excel multi-sheet**.
- Tambah pilihan gaya jawaban: **Detail**, **Ringkas**, dan **Management**.
- Mendukung **Streamlit Secrets** (`OPENAI_API_KEY`, `GEMINI_API_KEY`) sehingga key tidak perlu diketik ulang saat deployment cloud.
- AI Presentation juga otomatis membaca API key dari Streamlit Secrets jika tersedia.
- Pertanyaan historis menghitung inventory snapshot sesuai akhir periode yang ditanyakan, bukan selalu kondisi hari terakhir data.


## Update V2.2 — Google Gemini AI

- Tambah provider **Google Gemini** pada menu AI Presentation.
- Provider dapat dipilih: `Google Gemini` atau `OpenAI`.
- Gemini default menggunakan `gemini-3.7-flash`, dengan fallback stabil ke `gemini-3.6-flash`, `gemini-3.5-flash`, lalu `gemini-3.5-flash-lite` bila model terpilih tidak tersedia.
- Integrasi menggunakan package resmi `google-genai`; jika SDK tidak tersedia, aplikasi mempunyai HTTPS fallback langsung ke Gemini API.
- API key Gemini diinput melalui field password dan tidak ditulis ke cache/file.
- Error quota/API key dibuat ringkas tanpa traceback panjang.
- OpenAI dan Gemini memakai PowerPoint builder yang sama, sehingga angka slide tetap berasal dari analytics engine aplikasi.
- `setup_windows.bat` diperkuat agar memprioritaskan Python 3.11 di `%LocalAppData%\Programs\Python\Python311\python.exe`.
- Tambah `RUN_INDOKIDS.bat` sebagai launcher harian satu klik.

## Update V2.1 — AI runtime fix

- `run_app.bat` sekarang **selalu menggunakan `.venv` yang sama** dengan `setup_windows.bat`. Ini memperbaiki kasus Streamlit berjalan dari Python sistem sementara package `openai` terpasang di `.venv`.
- AI Presentation memiliki **HTTPS fallback langsung ke OpenAI Responses API** bila Python SDK `openai` belum tersedia, sehingga menu AI tidak langsung gagal hanya karena package SDK tidak ditemukan.
- Model API default diperbaiki menjadi **`gpt-5.6`**.
- Tambah `repair_ai_windows.bat` untuk memperbaiki dependency `openai` dan `python-pptx` tanpa reinstall seluruh project.
- Menu AI menampilkan status runtime: OpenAI SDK/HTTPS fallback dan kesiapan PowerPoint engine.

## Update V2

- Semua nominal revenue/value ditampilkan dalam format Indonesia, contoh: **`Rp. 1.234.567`**.
- SKU 360 memiliki filter **Inventory Status** sebelum memilih SKU.
- Pareto & Product Opportunity memiliki tabel khusus **Produk yang Perlu Ditingkatkan** dengan klasifikasi `GROW_REVENUE` dan `RECOVER_AVAILABILITY`, lengkap dengan download CSV.
- Tambahan menu **AI Presentation Generator** untuk menghasilkan insight management dan file PowerPoint `.pptx`.
- PowerPoint menggunakan angka yang dihitung aplikasi; OpenAI digunakan untuk narrative insight dan recommended actions, sehingga angka utama deck tidak dibuat oleh AI.

## AI Provider untuk AI Presentation

Menu **AI Presentation** mendukung dua provider:

### Google Gemini
- API key: Google AI Studio / Gemini API.
- Default: `gemini-3.7-flash`.
- Opsi lain: `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`.
- Auto-fallback hanya aktif untuk error model unavailable/deprecated; quota, API key, dan network error tidak disembunyikan.
- Parameter sampling lama (`temperature`, `top_p`, `top_k`) tidak dikirim ke Gemini 3.x.
- SDK: `google-genai`.

### OpenAI
- API key: OpenAI Platform.
- Model: `gpt-5.6`.
- ChatGPT Plus/Pro dan OpenAI API menggunakan billing terpisah.

API key hanya digunakan saat tombol Generate ditekan dan tidak ditulis ke cache Parquet atau file project.

Deck default terdiri dari:

1. Cover / Management Brief
2. Executive Overview
3. Target Performance
4. Pareto & Revenue Concentration
5. Products to Improve
6. Inventory Health
7. Recommended Actions
8. Management Focus

Raw transaction tidak dikirim ke OpenAI. Hanya KPI agregat, tren bulanan, ringkasan Pareto, top product opportunities, inventory summary, supplier/category summary, dan anomaly counts.

## Empat file input

### 1. Stock Awal — wajib
Kolom:

`sku,nama_barang,supplier,subdept,kel_barang,sub_kel,saldo_awal,hrg_beli,subtotal`

### 2. Kartu Stok — wajib
Kolom:

`kd_trx,tgl,sku,nama_barang,supplier,subdept,kel_barang,sub_kel,stock_in,stock_out,harga,subtotal,keterangan`

### 3. Histori Pembelian — opsional, sangat direkomendasikan
Kolom:

`tgl,no_faktur_beli,sku,harga_beli`

`hrg_beli` juga diterima dan otomatis dinormalisasi menjadi `harga_beli`.

### 4. Target Cabang — opsional
Kolom minimal:

`bulan,lokasi,target_omzet`

Format target numerik disarankan tanpa `Rp` atau pemisah ribuan. Aplikasi tetap mencoba membersihkan format `Rp756,695,582` bila ditemukan.

## Menjalankan di Windows

### Cara pertama — command prompt

1. Install **Python 3.11 atau 3.12**.
2. Buka Command Prompt pada folder aplikasi.
3. Jalankan:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

### Cara berikutnya

Setelah dependency terpasang, cukup double-click:

`RUN_INDOKIDS.bat`

**Penting:** gunakan `RUN_INDOKIDS.bat`, karena script ini menjalankan Streamlit dengan interpreter `.venv` yang sama dengan dependency aplikasi. Bila menu AI bermasalah setelah upgrade dari V2, jalankan `repair_ai_windows.bat` sekali.

Browser akan membuka aplikasi Streamlit lokal.

## Menjalankan di Linux / Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
./run_app.sh
```

## Formula inti

### Current Stock

`Opening Stock + cumulative Stock In - cumulative Stock Out`

### Net Sales

`Gross Sales - Sales Return`

### Estimated HPP priority

1. Last Purchase Price pada/sebelum tanggal transaksi.
2. Last cost-bearing stock-in (Pre Receive / Transfer In / Adjustment In / Opname In).
3. Opening Stock cost.
4. First future known cost sebagai fallback SKU baru.
5. Jika tetap tidak ada: `UNRESOLVED` dan ditampilkan di Anomaly Center.

### Target Chase

- `Achievement = Actual MTD / Target`
- `Required Daily Sales = Remaining Target / Remaining Calendar Days`
- `Projected Month End = Daily Run Rate × Days in Month`
- `Projected Gap = max(Target - Projected Month End, 0)`

### Dynamic Pareto

- **Core 20**: top 20% SKU berdasarkan revenue.
- **A80**: jumlah minimum SKU sampai cumulative revenue ≥80%.
- **Opportunity**: SKU di antara Core 20 dan A80.
- **Long Tail**: SKU sesudah A80.

Aplikasi tidak memaksa asumsi bahwa 20% SKU selalu menghasilkan 80% revenue.

### Inventory statistical bands

Baseline menggunakan bulan lengkap saja. SKU baru dihitung mulai bulan aktivasi/first stock-in.

- `Slow Threshold = μ + 1σ`
- `Dead Threshold = μ + 3σ`
- `Overstock Threshold = μ + 6σ`

Status dibuat **mutually exclusive**:

`Normal → Slow → Dead → Overstock`

Special status: `Negative`, `Stockout`, `Zero Stock`, dan `No Sales`.

## Performance / cache

- Streamlit `cache_data` digunakan agar upload yang sama tidak diproses ulang pada setiap rerun.
- Setelah proses, aplikasi mencoba membuat cache Parquet di folder `.cache/<fingerprint>/`.
- Bila Parquet tidak tersedia, aplikasi tetap jalan menggunakan cache memori Streamlit.

## Catatan HPP

Gross Profit dan Gross Margin di aplikasi adalah **Estimated Management HPP**, bukan pengganti costing akuntansi FIFO/moving-average resmi. Menu Profitability selalu menampilkan sumber HPP dan missing coverage agar hasil transparan.

## Folder project

```text
indokids_branch_command_center/
├── app.py
├── requirements.txt
├── run_app.bat
├── run_app.sh
├── .streamlit/config.toml
├── src/
│   ├── ai_presentation.py
│   ├── analytics.py
│   ├── cache_store.py
│   ├── config.py
│   ├── hpp.py
│   ├── insights.py
│   ├── io.py
│   ├── pipeline.py
│   ├── transform.py
│   └── utils.py
├── templates/
└── tests/
```

## Filter Produk V2.5

Filter produk menggunakan mekanisme **draft → apply**. Pilih Supplier/Subdept/Kel Barang/Sub Kel di sidebar, lalu klik **Proses Data Sesuai Filter**. Halaman detail baru menggunakan filter setelah tombol tersebut diklik. Gunakan **Reset Filter** untuk kembali ke seluruh data. Jika filter aktif, menu Ask Anything by AI otomatis menggunakan scope **Ikuti Filter Produk Sidebar**.


## V2.8 — Advanced AI Presentation Studio
See `README_V2_8.md` for the complete workflow. AI Presentation now supports Presentation Brief, audience-specific focus, dynamic slide planning, editable slide plan, Executive/Standard/Deep Dive depth, and dynamic 7–20 slide PPTX generation.
