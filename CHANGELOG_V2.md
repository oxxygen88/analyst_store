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
