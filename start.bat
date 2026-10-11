@echo off
setlocal EnableExtensions
title ResumeAid Launcher
cd /d "%~dp0"

echo.
echo  ==============================
echo    ResumeAid - starting up
echo  ==============================
echo.

set "BACKEND=%~dp0backend"
set "VENV=%BACKEND%\.venv"
set "PORT=5000"

rem ---------- 1. Find Python ----------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH" during setup.
    pause
    exit /b 1
)

rem ---------- 2. Pick an environment ----------
rem Prefer the project venv. If none exists, reuse the global Python when it
rem already has the dependencies - avoids re-downloading torch (~2 GB).
set "PY=python"
if exist "%VENV%\Scripts\python.exe" (
    set "PY=%VENV%\Scripts\python.exe"
    echo [OK] Using project virtual environment.
    goto :check_deps
)

python -c "import flask, flask_cors, sentence_transformers, spacy, pdfplumber, docx, bs4, requests" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Using global Python - dependencies already installed.
    rem python-dotenv is small but required to load backend\.env
    python -c "import dotenv" >nul 2>&1
    if errorlevel 1 (
        echo [..] Installing python-dotenv ...
        python -m pip install -q python-dotenv
    )
    goto :check_model
)

echo [..] Creating virtual environment in backend\.venv ...
python -m venv "%VENV%"
if errorlevel 1 (
    echo [ERROR] Could not create virtual environment.
    pause
    exit /b 1
)
set "PY=%VENV%\Scripts\python.exe"

:check_deps
rem Reinstall only when requirements.txt changed since the last install.
set "STAMP=%VENV%\.requirements.installed"
fc /b "%BACKEND%\requirements.txt" "%STAMP%" >nul 2>&1
if not errorlevel 1 goto :check_model

echo [..] Installing dependencies - first run can take several minutes...
"%PY%" -m pip install --upgrade pip -q
"%PY%" -m pip install -r "%BACKEND%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See messages above.
    pause
    exit /b 1
)
copy /y "%BACKEND%\requirements.txt" "%STAMP%" >nul

:check_model
rem ---------- 3. spaCy language model ----------
"%PY%" -c "import en_core_web_sm" >nul 2>&1
if errorlevel 1 (
    echo [..] Downloading spaCy model en_core_web_sm ...
    "%PY%" -m spacy download en_core_web_sm
    if errorlevel 1 (
        echo [ERROR] Could not download the spaCy model.
        pause
        exit /b 1
    )
)

rem ---------- 4. .env file ----------
if not exist "%BACKEND%\.env" (
    if exist "%BACKEND%\.env.example" (
        copy "%BACKEND%\.env.example" "%BACKEND%\.env" >nul
        echo [OK] Created backend\.env from template - add Adzuna keys there if you have them.
    )
)

rem ---------- 5. Start backend ----------
rem Skip if something is already serving the API on this port.
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 (
    echo [OK] Backend already running on port %PORT%.
    goto :open_frontend
)

echo [..] Starting backend API in a new window...
start "ResumeAid API" /d "%BACKEND%" cmd /k ""%PY%" api.py"

echo [..] Waiting for the API to come online - loading the AI model...
set /a TRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 goto :ready
set /a TRIES+=1
if %TRIES% lss 90 goto :wait_loop

echo [WARN] The API didn't respond within 3 minutes.
echo        Check the "ResumeAid API" window for errors. Opening the app anyway...
goto :open_frontend

:ready
echo [OK] Backend is up at http://127.0.0.1:%PORT%

:open_frontend
rem ---------- 6. Open frontend ----------
echo [..] Opening ResumeAid in your browser...
start "" "%~dp0index.html"

echo.
echo  Done! Close the "ResumeAid API" window to stop the server.
echo.
timeout /t 5 >nul
endlocal
