# Validation Report — IDK-ATP sample files

Aplikasi divalidasi menggunakan file Antapani yang diberikan pada percakapan.

## Coverage

- Stock awal: 66.520 rows.
- Kartu stok: 173.716 rows.
- Histori pembelian: 5.100 rows.
- Target: 12 bulan.
- Transaction coverage: 1 Jan 2026 sampai 22 Aug 2026.

## August MTD check (through 22 Aug 2026)

- Target: Rp. 605.102.070
- Actual Net Sales: sekitar Rp. 358.597.985
- Achievement: sekitar 59,26%
- Required Daily Sales for 9 calendar days remaining: sekitar Rp. 27,39 juta/hari
- Projected Month End at current daily run-rate: sekitar Rp. 505,30 juta
- Projected Gap: sekitar Rp. 99,80 juta

## Dynamic Pareto check — August MTD

- Active selling SKU: 3.055
- Core 20% SKU: 611
- Core 20 revenue share: sekitar 58,50%
- A80 tercapai pada sekitar 41,90% active selling SKU
- Opportunity group: 669 SKU

Temuan ini memvalidasi alasan penggunaan **Dynamic Pareto** daripada memaksa asumsi 20% SKU selalu menyumbang 80% revenue.

## HPP source coverage — full transaction data

Mayoritas commercial lines berhasil di-resolve melalui:

- cost-bearing stock in / transfer / pre-receive,
- last purchase history,
- opening stock,
- fallback first known cost untuk sebagian kecil SKU baru.

Unresolved HPP tetap ditampilkan eksplisit di Anomaly Center dan tidak disembunyikan.

## Performance after vectorization

Pada environment pengujian internal dengan dataset Antapani:

- HPP resolution: sekitar sub-second setelah load/normalization.
- Inventory health snapshot: sekitar <1 second.
- Pareto + Opportunity scoring: sekitar <1 second.

Angka aktual di PC pengguna dapat berbeda berdasarkan CPU, RAM, versi Python, dan filesystem.


## V2 regression check

- Currency formatter check: `1234567` → `Rp. 1.234.567`.
- SKU 360 Status filter: implemented before SKU selection.
- Product improvement action table: implemented with downloadable raw numeric CSV.
- AI presentation context is aggregate-only; raw transaction rows are excluded.
- PPTX builder validated against the IDK-ATP dataset using mock AI insight content.
- Generated deck: 8 slides, PowerPoint file opens/renders successfully.
- Core automated tests after V2 update: **5/5 passed**.


## V2.6 Gemini Compatibility Validation

- Gemini default model updated to `gemini-3.7-flash`.
- Auto-fallback model-unavailable path tested: 3.7 → 3.6.
- Quota errors verified to stop immediately without fallback masking.
- Flash-Lite selection verified not to escalate to higher-cost models.
- Gemini 3.x requests no longer send deprecated sampling parameters.
- Full regression suite: **14/14 passed**.
