@echo off
setlocal EnableExtensions
title ResumeAid Live Server (ngrok Static Tunnel)
cd /d "%~dp0"

echo.
echo  ====================================================
echo    ResumeAid - Live Public Server (ngrok Tunnel)
echo  ====================================================
echo.

set "BACKEND=%~dp0backend"
set "ROOT_VENV=%~dp0.venv"
set "BACKEND_VENV=%BACKEND%\.venv"
set "PORT=5000"
set "NGROK_DOMAIN=undoing-eligible-canal.ngrok-free.dev"

rem ---------- 1. Find or install ngrok ----------
set "NGROK_EXE=ngrok"
where ngrok >nul 2>&1
if not errorlevel 1 goto :pick_python

if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe" (
    set "NGROK_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe"
    goto :pick_python
)

echo [..] ngrok not found. Installing via winget...
winget install --id Ngrok.Ngrok --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe" (
    set "NGROK_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe"
    goto :pick_python
)
where ngrok >nul 2>&1
if not errorlevel 1 goto :pick_python

echo.
echo [ERROR] ngrok is not installed or not on PATH yet.
echo         Please close this window and double-click start_live.bat again,
echo         or install ngrok from https://ngrok.com/download
echo.
pause
exit /b 1

:pick_python
rem ---------- 2. Pick Python ----------
set "PY=python"
if exist "%ROOT_VENV%\Scripts\python.exe" (
    set "PY=%ROOT_VENV%\Scripts\python.exe"
    echo [OK] Using virtual environment (.venv).
) else if exist "%BACKEND_VENV%\Scripts\python.exe" (
    set "PY=%BACKEND_VENV%\Scripts\python.exe"
    echo [OK] Using virtual environment (backend\.venv).
) else (
    echo [OK] Using system Python.
)

rem ---------- 3. Start backend API if not already running ----------
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 (
    echo [OK] Backend already running on port %PORT%.
    goto :start_tunnel
)

echo [..] Starting ResumeAid backend API on port %PORT%...
start "ResumeAid API" /d "%BACKEND%" cmd /k ""%PY%" api.py"

echo [..] Waiting for the AI model to finish loading (takes ~15-30 seconds)...
set /a TRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 goto :start_tunnel
set /a TRIES+=1
if %TRIES% lss 90 goto :wait_loop

echo [WARN] Backend took longer than expected; check the "ResumeAid API" window for errors.
pause

:start_tunnel
echo.
echo  =====================================================================
echo   [OK] Backend is online at http://127.0.0.1:%PORT%
echo   [..] Starting permanent ngrok tunnel:
echo        https://%NGROK_DOMAIN%
echo  =====================================================================
echo.
"%NGROK_EXE%" http --url=%NGROK_DOMAIN% %PORT%

if errorlevel 1 (
    echo.
    echo  =====================================================================
    echo   [!] ngrok exited with an error (see message above).
    echo       If it says "authentication failed" or "authtoken":
    echo       Copy your token from: https://dashboard.ngrok.com/get-started/your-authtoken
    echo  =====================================================================
    echo.
    set /p NGROK_TOKEN="Paste your ngrok authtoken here (or press Enter to exit): "
    if not "%NGROK_TOKEN%"=="" (
        "%NGROK_EXE%" config add-authtoken %NGROK_TOKEN%
        echo.
        echo [OK] Token saved! Restarting tunnel...
        echo.
        goto :start_tunnel
    )
)

echo.
pause
endlocal
