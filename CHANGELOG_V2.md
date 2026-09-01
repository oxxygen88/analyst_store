# V2.8 — Advanced AI Presentation Studio

## New presentation workflow
- Presentation Brief: objective, focus areas, depth, and mandatory user discussion points.
- Audience-aware default focus for Management/Owner, Buyer & Inventory, and Store Operations.
- AI Slide Plan generation before final deck generation.
- Editable slide plan using Streamlit data editor: include/exclude, order, slide type, title, objective, emphasis, and dynamic rows.
- Recommended local slide plan fallback without using an extra AI call.
- Dynamic deck depth: Executive 7–9, Standard 10–14, Deep Dive 15–20 slides.
- Dynamic slide library: Executive, Target, Sales Growth, Gap Diagnosis, Pareto, Pareto Migration, Product Opportunity, Stockout Recovery, Inventory Health, Inventory Capital, Profitability, Supplier, Category, Transfer, Purchase, Anomaly, 30-Day Action Plan, Closing.

## Analytics fact-pack enhancements
- Current MTD vs previous-month comparable-period commercial comparison.
- Daily sales fact-pack.
- Inventory capital by status.
- Stockout/revenue recovery candidate list.
- Pareto migration and product momentum.
- Purchase by supplier and transfer flow summaries.
- Profitability leaders.

## AI governance
- AI cannot invent numeric facts or unsupported causal explanations.
- User brief is mandatory only when supported by application data.
- Facts, data-supported interpretation, hypothesis, and recommendation are explicitly separated in the prompt.
- Raw transaction rows are not sent directly to the AI provider.
- Charts, tables, KPI cards, and monetary values are generated locally from the analytics engine.

## Validation
- 20/20 automated tests passed.
- Smoke-tested against the real IDK-ATP dataset through 22 Aug 2026.
- Deep Dive generated 19 slides successfully.

---

# V2.6 — Gemini 3.x Compatibility & Auto Fallback

- Replaced deprecated/retired Gemini 2.5 selections with stable Gemini 3.x IDs.
- Default model: `gemini-3.7-flash`.
- Stable fallback chain: `gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash → gemini-3.5-flash-lite`.
- Auto-fallback triggers only for model unavailable/deprecated/404 NOT_FOUND errors.
- Quota, API key/auth, malformed request, and network errors are not masked by fallback.
- Removed deprecated Gemini 3.x sampling parameters (`temperature`, `top_p`, `top_k`).
- Added stale Streamlit session-state migration from old 2.5 model IDs to the new default.
- Updated both Ask Anything by AI and AI Presentation Generator.
- Regression tests: 14/14 passed.

# V2.5 — Explicit Filter Processing

- Added **Proses Data Sesuai Filter** button in the sidebar.
- Sidebar selections are now draft filters until explicitly processed.
- Added **Reset Filter** button and active-filter indicator.
- Added active SKU scope count.
- Detail pages use only the committed/applied filter state.
- Ask Anything by AI automatically switches to **Ikuti Filter Produk Sidebar** after a product filter is processed.
- Clearing filters automatically returns Ask AI to **Seluruh Cabang**.
- AI supporting-data state is cleared when filter scope changes to prevent stale answers/tables.
- Regression tests: 11/11 passed.

# V2.4 — Ask Anything by AI

- Added `Ask Anything by AI` menu with Gemini/OpenAI provider selection.
- Added deterministic local fact-pack/query routing for sales, target, product, inventory, Pareto, movement, supplier/category, profitability, and anomalies.
- Raw transaction rows are not sent directly to AI providers.
- Added chat history/follow-up support and previous-period scope inheritance.
- Added supporting-data preview and multi-sheet Excel download for the last AI answer.
- Added Streamlit Secrets support for `OPENAI_API_KEY` and `GEMINI_API_KEY` in both Ask AI and AI Presentation menus.
- Historical inventory questions use the requested period-end snapshot.
- Regression suite expanded to 11 tests; all passing.

# V2.3 — Streamlit Cloud Compatibility Hotfix

- Updated PyArrow requirement from `pyarrow>=16,<22` to `pyarrow>=23,<26`.
- Fixes Streamlit Community Cloud deployment on Python 3.14 where PyArrow 21 was built from source and failed because CMake was unavailable.
- Added `STREAMLIT_CLOUD_DEPLOY.md` with deployment and Python-version guidance.

# Changelog V2.2

## Google Gemini AI
- Tambah provider Google Gemini di AI Presentation.
- Model default `gemini-2.5-flash`; opsi `gemini-2.5-pro` dan `gemini-2.5-flash-lite`.
- Integrasi `google-genai` + HTTPS fallback.
- Provider selector: Gemini atau OpenAI.
- Error quota/API key diringkas tanpa traceback panjang.
- Setup Windows memprioritaskan Python 3.11 lokal dan memasang OpenAI + Gemini + python-pptx.
- Tambah `RUN_INDOKIDS.bat` sebagai launcher satu klik.

# Changelog V2.1

## Hotfix AI Menu
- Memperbaiki mismatch Python environment: `setup_windows.bat` menginstall dependency ke `.venv`, sementara `run_app.bat` V2 sebelumnya menjalankan `python -m streamlit` dari Python sistem.
- `run_app.bat` sekarang menjalankan `.venv\Scripts\python.exe -m streamlit run app.py`.
- `setup_windows.bat` memakai `.venv\Scripts\python.exe -m pip` untuk seluruh instalasi dan melakukan dependency verification.
- Tambah `repair_ai_windows.bat`.
- Bila package `openai` tidak tersedia, AI insight otomatis memakai HTTPS fallback ke OpenAI Responses API.
- Model diperbaiki dari nama eksperimen `gpt-5.6-luna/terra/sol` menjadi API model `gpt-5.6`.
- Runtime status AI ditampilkan di menu Presentation.

## 1. Currency formatting
- Semua helper Rupiah menggunakan format `Rp. 1.234.567`.
- Tabel penting memakai formatter Rupiah dan persen tanpa mengubah nilai numerik file download.
- Chart revenue/stock value memakai prefix `Rp.` dan separator Indonesia.

## 2. SKU 360 Status Filter
- Tambah multi-select `Filter Status` sebelum pemilihan SKU.
- Label SKU menampilkan status aktif, contoh `[OVERSTOCK]`.

## 3. Products to Improve
- Tambah tabel khusus pada Pareto & Product Opportunity.
- Focus Type: `GROW_REVENUE` dan `RECOVER_AVAILABILITY`.
- Filter inventory status, focus type, dan minimum Opportunity Score.
- Download CSV full table.

## 4. AI Presentation Generator
- Menu baru `AI Presentation`.
- OpenAI Responses API integration.
- API key password input; key tidak ditulis ke cache/file.
- Model selectable: gpt-5.6-luna / terra / sol.
- Audience dan bahasa presentasi selectable.
- AI menerima data agregat saja; raw transaction tidak dikirim.
- PowerPoint 8 slide dibangun lokal dengan python-pptx.
- Angka utama slide berasal dari analytics engine; AI membuat insight/rekomendasi.

## 5. Dependencies
- openai
- python-pptx

## Validation
- Existing + V2 unit tests: 5/5 passed.
- PPTX builder tested using full IDK-ATP dataset and visually rendered successfully.

## V2.7 — Flexible Branch Schema

- `sub_kel` sekarang **opsional** pada Stock Awal dan Kartu Stok.
- Jika source tidak memiliki `sub_kel`, engine menambahkan placeholder internal `(Tidak tersedia)` tanpa mengarang klasifikasi baru.
- Filter **Sub Kel** otomatis disembunyikan jika cabang memang tidak menyediakan level tersebut.
- Category & Supplier hanya menawarkan dimensi `sub_kel` bila data nyata tersedia.
- SKU 360 tidak menampilkan hierarchy `sub_kel` palsu.
- Menambahkan parser tanggal adaptif untuk export POS yang hanya mempunyai `tgl` berbentuk `MM:SS.0`.
- Untuk format tersebut, tanggal kalender direkonstruksi dari `kd_trx` (YYMMDD) dan `time_available=False`.
- Hourly Sales otomatis dinonaktifkan jika jam transaksi tidak dapat dipercaya.
- Reference transaksi di luar tahun analisis ditandai `OUTSIDE_ANALYSIS_YEAR`, tidak dihitung sebagai movement tahun berjalan, dan muncul di Data & Anomaly Center.
- Smoke test NRM-HJL 2026 berhasil: 877.720 baris transaksi, 61.213 SKU master, coverage 1 Jan–25 Agu 2026.
