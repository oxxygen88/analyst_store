@echo off
setlocal EnableExtensions
title INDOKIDS Branch Command Center
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%~dp0app.py" (
    echo [ERROR] app.py tidak ditemukan.
    echo Pastikan RUN_INDOKIDS.bat berada di folder yang sama dengan app.py.
    pause
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo [INFO] .venv belum tersedia. Menjalankan setup...
    call "%~dp0setup_windows.bat"
    if errorlevel 1 exit /b 1
)

echo [CHECK] Memeriksa dependency utama...
"%VENV_PY%" -c "import streamlit, pptx; print('Dependency utama OK')" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Dependency belum lengkap. Menjalankan setup...
    call "%~dp0setup_windows.bat"
    if errorlevel 1 exit /b 1
)

echo.
echo ============================================================
echo  INDOKIDS Branch Command Center
echo  Local URL: http://localhost:8501
echo  Tekan CTRL+C untuk menghentikan aplikasi.
echo ============================================================
echo.

"%VENV_PY%" -m streamlit run app.py
pause
