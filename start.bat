@echo off
REM Simple batch file to start both servers with one command
echo.
echo ========================================
echo  LSTM Phishing Detection App
echo ========================================
echo.
echo Starting both Python API and Node.js server...
echo.
echo [Python API]  http://localhost:5000
echo [Web App]     http://localhost:3000
echo.
echo Press Ctrl+C to stop both servers
echo.

cd /d "%~dp0"
npm run dev

pause
