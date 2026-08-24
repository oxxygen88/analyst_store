@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title INDOKIDS Branch Command Center - Setup

echo ============================================================
echo  INDOKIDS Branch Command Center - Windows Setup
echo ============================================================
echo.

set "PY311=%LocalAppData%\Programs\Python\Python311\python.exe"
set "PYBOOT="

if exist "%PY311%" (
  set "PYBOOT="%PY311%""
  echo [INFO] Python dipilih: %PY311%
) else (
  py -3.11 --version >nul 2>&1
  if not errorlevel 1 (
    set "PYBOOT=py -3.11"
    echo [INFO] Python dipilih: py -3.11
  ) else (
    where python >nul 2>&1
    if not errorlevel 1 (
      set "PYBOOT=python"
      echo [INFO] Python dipilih dari PATH.
    )
  )
)

if "%PYBOOT%"=="" (
  echo [ERROR] Python 3.11 tidak ditemukan.
  echo Install Python 3.11 64-bit dari python.org lalu jalankan kembali.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Membuat virtual environment .venv ...
  %PYBOOT% -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/4] Virtual environment sudah ada.
)

echo [2/4] Upgrade pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/4] Install/update dependency ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/4] Verifikasi Streamlit, OpenAI, Gemini, dan PowerPoint ...
".venv\Scripts\python.exe" -c "import streamlit, openai, pptx; from google import genai; print('OK - streamlit', streamlit.__version__, '| openai', openai.__version__, '| Gemini SDK siap | python-pptx siap')"
if errorlevel 1 goto :error

echo.
echo Setup selesai.
echo Selanjutnya cukup double-click RUN_INDOKIDS.bat.
pause
exit /b 0

:error
echo.
echo SETUP GAGAL.
echo Pastikan internet aktif dan Python 3.11 dapat diakses.
echo Jika ada error, kirim screenshot baris error di atas.
pause
exit /b 1
