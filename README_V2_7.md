# V2.7 Flexible Branch Schema

V2.7 membuat input product hierarchy fleksibel antar cabang.

## Format yang didukung

### Dengan sub_kel
Stock Awal:
`sku,nama_barang,supplier,subdept,kel_barang,sub_kel,saldo_awal,hrg_beli,subtotal`

Kartu Stok:
`kd_trx,tgl,sku,nama_barang,supplier,subdept,kel_barang,sub_kel,stock_in,stock_out,harga,subtotal,keterangan`

### Tanpa sub_kel
Stock Awal:
`sku,nama_barang,supplier,subdept,kel_barang,saldo_awal,hrg_beli,subtotal`

Kartu Stok:
`kd_trx,tgl,sku,nama_barang,supplier,subdept,kel_barang,stock_in,stock_out,harga,subtotal,keterangan`

Keduanya diproses oleh engine yang sama. Jika `sub_kel` tidak ada, aplikasi tidak mengarang nilai subkategori; filter dan analisis berjalan sampai `kel_barang`.

## Format tanggal NRM-HJL

File contoh NRM-HJL mempunyai `tgl` seperti `16:44.0` sampai `59:59.0`, sehingga kolom tersebut tidak mengandung tanggal kalender lengkap. V2.7 mengambil YYMMDD dari `kd_trx`, misalnya:

`THJ2601010001C` -> `2026-01-01`

Jam tidak direkonstruksi karena source tidak menyediakannya secara terpercaya. Karena itu Hourly Sales dinonaktifkan untuk dataset tersebut.

## Update Streamlit Cloud

Untuk existing V2.6, replace file dari hotfix V2.7:
- `app.py`
- `src/config.py`
- `src/io.py`
- `src/transform.py`
- `src/analytics.py`

`requirements.txt` tidak berubah.
