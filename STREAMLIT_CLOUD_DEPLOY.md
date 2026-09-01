# INDOKIDS Branch Command Center — Streamlit Community Cloud

## Hotfix V2.3 Cloud

### Akar masalah deployment sebelumnya
Streamlit Cloud menjalankan environment Python 3.14.7, sedangkan `requirements.txt` lama membatasi `pyarrow>=16,<22`. Resolver memilih `pyarrow==21.0.0`, yang tidak menyediakan binary wheel untuk Python 3.14. Akibatnya Cloud mencoba build PyArrow dari source dan gagal pada CMake.

### Perbaikan V2.3
Dependency diubah menjadi:

```text
pyarrow>=23,<26
```

PyArrow 23+ menyediakan wheel untuk Python 3.14 sehingga tidak perlu build C++/CMake pada Streamlit Community Cloud.

## Cara update repository yang SUDAH ada
1. Replace `requirements.txt` repository GitHub dengan file V2.3.
2. Commit dan push ke branch yang dipakai Streamlit (`main`).
3. Streamlit Community Cloud akan mendeteksi perubahan dependency dan melakukan redeploy penuh.
4. Buka **Manage app > Logs** dan pastikan PyArrow didownload sebagai `.whl`, bukan `.tar.gz`.

## Pilihan versi Python
Aplikasi dikembangkan lokal menggunakan Python 3.11. Untuk konsistensi maksimum, saat membuat deployment baru di Streamlit Community Cloud:

1. Pilih **Advanced settings**.
2. Pilih Python **3.11** atau **3.12**.
3. Deploy.

Jika app yang sudah dibuat sedang menggunakan Python 3.14, versi Python tidak dapat diubah in-place. Anda dapat tetap menggunakan Python 3.14 dengan requirements V2.3 ini, atau delete + redeploy bila ingin menyamakan versi dengan lokal.

## File penting di root repository
Pastikan struktur GitHub seperti berikut:

```text
app.py
requirements.txt
src/
.streamlit/config.toml
templates/
```

Jangan meng-upload hanya isi `src/`.

## API key di Streamlit Cloud
Versi sekarang tetap memungkinkan API key dimasukkan melalui UI dengan field password. Untuk deployment internal/production, lebih aman menyimpan key melalui Streamlit **Secrets** dan tidak commit key ke GitHub.

Contoh secrets:

```toml
OPENAI_API_KEY = "..."
GEMINI_API_KEY = "..."
```

Jangan commit file `.streamlit/secrets.toml` yang berisi key asli ke repository public.


## Ask Anything by AI + API Secrets
Untuk deployment Streamlit Cloud, disarankan menyimpan API key di **App > Settings > Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "AIza..."
```

Menu **Ask Anything by AI** dan **AI Presentation** akan otomatis memakai secret tersebut. Jika secret tidak tersedia, aplikasi tetap menampilkan field password manual. Jangan commit key asli ke GitHub.

Ask Anything by AI tidak mengirim raw transaction penuh ke provider. Aplikasi membangun fact pack dari kalkulasi lokal dan hanya mengirim tabel agregat yang relevan dengan pertanyaan.


## V2.8 update
No dependency change. Replace `app.py` and `src/ai_presentation.py`, then commit/push. Existing Streamlit Secrets `GEMINI_API_KEY` / `OPENAI_API_KEY` continue to work.
