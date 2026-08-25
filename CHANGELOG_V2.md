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
