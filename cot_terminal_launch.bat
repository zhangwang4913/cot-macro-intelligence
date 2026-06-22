@echo off
cd /d "%~dp0"

echo Stopping any existing server on port 5050...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5050 "') do taskkill /PID %%a /F >nul 2>&1

echo Starting COT Macro Intelligence Dashboard...
echo Browser will open at http://localhost:5050
echo Close this window to stop the server.
echo.
py dashboard_server.py
pause
