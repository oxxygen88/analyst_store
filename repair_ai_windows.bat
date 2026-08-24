@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title INDOKIDS - Repair AI Dependencies

echo ============================================================
echo  Repair AI Dependencies - OpenAI + Google Gemini + PPTX
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] .venv belum ada. Menjalankan setup lengkap...
  call setup_windows.bat
  exit /b %errorlevel%
)

echo [1/2] Install/update AI dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade "openai>=1.100,<3" "google-genai>=1.0,<2" "python-pptx>=1.0,<2"
if errorlevel 1 goto :error

echo [2/2] Verifikasi...
".venv\Scripts\python.exe" -c "import openai, pptx; from google import genai; print('AI dependency OK - OpenAI', openai.__version__, '| Gemini SDK siap | PPTX siap')"
if errorlevel 1 goto :error

echo.
echo Repair selesai.
echo Tutup aplikasi lama lalu buka kembali melalui RUN_INDOKIDS.bat.
pause
exit /b 0

:error
echo.
echo Repair gagal. Pastikan internet aktif.
pause
exit /b 1
