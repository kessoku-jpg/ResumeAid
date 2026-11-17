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
set "VENV=%BACKEND%\.venv"
set "PORT=5000"
set "NGROK_DOMAIN=undoing-eligible-canal.ngrok-free.dev"

rem ---------- 1. Check ngrok ----------
where ngrok >nul 2>&1
if errorlevel 1 (
    echo [..] ngrok not found on PATH. Installing via winget...
    winget install --id Ngrok.Ngrok --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Could not install ngrok automatically.
        echo         Run: winget install Ngrok.Ngrok
        pause
        exit /b 1
    )
    echo [OK] ngrok installed. If PATH is not refreshed yet, reopen this script once.
)

rem ---------- 2. Pick Python ----------
set "PY=python"
if exist "%VENV%\Scripts\python.exe" (
    set "PY=%VENV%\Scripts\python.exe"
)

rem ---------- 3. Start backend API if not already running ----------
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 (
    echo [OK] Backend already running on port %PORT%.
    goto :start_tunnel
)

echo [..] Starting ResumeAid backend API on port %PORT%...
start "ResumeAid API" /d "%BACKEND%" cmd /k ""%PY%" api.py"

echo [..] Waiting for the AI model to finish loading...
set /a TRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 goto :start_tunnel
set /a TRIES+=1
if %TRIES% lss 90 goto :wait_loop

echo [WARN] Backend took longer than expected; starting tunnel anyway...

:start_tunnel
echo.
echo  =====================================================================
echo   [OK] Backend is online at http://127.0.0.1:%PORT%
echo   [..] Starting permanent ngrok tunnel:
echo        https://%NGROK_DOMAIN%
echo  =====================================================================
echo.
ngrok http --url=%NGROK_DOMAIN% %PORT%
endlocal
