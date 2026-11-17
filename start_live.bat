@echo off
setlocal EnableExtensions
title ResumeAid Live Server (Cloudflare Tunnel)
cd /d "%~dp0"

echo.
echo  ====================================================
echo    ResumeAid - Live Public Server (Cloudflare Tunnel)
echo  ====================================================
echo.

set "BACKEND=%~dp0backend"
set "VENV=%BACKEND%\.venv"
set "PORT=5000"

rem ---------- 1. Check cloudflared ----------
where cloudflared >nul 2>&1
if errorlevel 1 (
    echo [..] cloudflared not found on PATH. Installing via winget...
    winget install --id Cloudflare.cloudflared --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Could not install cloudflared automatically.
        echo         Run: winget install Cloudflare.cloudflared
        pause
        exit /b 1
    )
    echo [OK] cloudflared installed. If PATH is not refreshed yet, reopen this script once.
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
echo   [..] Starting Cloudflare HTTPS Tunnel...
echo.
echo   HOW TO ACCESS FROM YOUR LAPTOP OR PHONE:
echo   1. Copy the "https://xxxx.trycloudflare.com" URL printed below.
echo   2. Open it directly on your laptop, OR open your Vercel site as:
echo      https://YOUR-SITE.vercel.app/?api=https://xxxx.trycloudflare.com
echo  =====================================================================
echo.
cloudflared tunnel --url http://127.0.0.1:%PORT%
endlocal
