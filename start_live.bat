@echo off
setlocal EnableExtensions
title ResumeAid Live Server - ngrok Static Tunnel
cd /d "%~dp0"

echo.
echo  ====================================================
echo    ResumeAid - Live Public Server [ngrok Tunnel]
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

if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe" set "NGROK_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe" goto :pick_python

echo [..] ngrok not found. Installing via winget...
winget install --id Ngrok.Ngrok --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe" set "NGROK_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ngrok.exe" goto :pick_python

where ngrok >nul 2>&1
if not errorlevel 1 goto :pick_python

echo.
echo [ERROR] ngrok could not be found automatically.
echo         Please close this window and open start_live.bat again,
echo         or download ngrok from https://ngrok.com/download
echo.
pause
exit /b 1

:pick_python
rem ---------- 2. Pick Python ----------
set "PY=python"
if not exist "%ROOT_VENV%\Scripts\python.exe" goto :check_backend_venv
set "PY=%ROOT_VENV%\Scripts\python.exe"
echo [OK] Using virtual environment: .venv
goto :check_running

:check_backend_venv
if not exist "%BACKEND_VENV%\Scripts\python.exe" goto :use_system_python
set "PY=%BACKEND_VENV%\Scripts\python.exe"
echo [OK] Using virtual environment: backend\.venv
goto :check_running

:use_system_python
echo [OK] Using system Python.

:check_running
rem ---------- 3. Start backend API if not already running ----------
curl -s -o nul -f http://127.0.0.1:%PORT%/api/health >nul 2>&1
if not errorlevel 1 goto :backend_already_up

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

echo [WARN] Backend took longer than expected. Check the "ResumeAid API" window.
pause
goto :start_tunnel

:backend_already_up
echo [OK] Backend already running on port %PORT%.

:start_tunnel
echo.
echo  =====================================================================
echo   [OK] Backend is online at http://127.0.0.1:%PORT%
echo   [..] Starting permanent ngrok tunnel:
echo        https://%NGROK_DOMAIN%
echo  =====================================================================
echo.
"%NGROK_EXE%" http --domain=%NGROK_DOMAIN% %PORT%
if errorlevel 1 goto :ngrok_error
goto :end

:ngrok_error
echo.
echo  =====================================================================
echo   [!] ngrok exited with an error.
echo       If it says "authentication failed" or "authtoken":
echo       Copy your token from: https://dashboard.ngrok.com/get-started/your-authtoken
echo  =====================================================================
echo.
set "NGROK_TOKEN="
set /p NGROK_TOKEN="Paste your ngrok authtoken here and press Enter: "
if "%NGROK_TOKEN%"=="" goto :end
"%NGROK_EXE%" config add-authtoken %NGROK_TOKEN%
echo.
echo [OK] Token saved! Restarting tunnel...
echo.
goto :start_tunnel

:end
echo.
pause
endlocal
