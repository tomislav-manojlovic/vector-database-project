@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo.
    echo [GRESKA] Nije pronadjen .venv u ovom folderu.
    echo UI ne zahteva nove pakete, ali koristi postojeci Python iz projekta.
    echo Otvori PowerShell u ovom folderu i jednom pokreni:
    echo.
    echo   py -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Pokrecem Qdrant Image Search UI...
"%PYTHON_EXE%" ui\server.py

if errorlevel 1 (
    echo.
    echo UI se zaustavio zbog greske. Proveri da li Docker Desktop i Qdrant rade.
    pause
)
