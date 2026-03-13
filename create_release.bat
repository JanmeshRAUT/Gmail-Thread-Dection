@echo off
REM Create model release package

setlocal enabledelayedexpansion

set MODEL_DIR=ml\models
set PACKAGE_DIR=lstm_model_release
set VERSION=1.0

if exist %PACKAGE_DIR% (
    echo Removing old package directory...
    rmdir /s /q %PACKAGE_DIR%
)

echo Creating release package...
mkdir %PACKAGE_DIR%

REM Copy model files
echo Copying model files...
copy "%MODEL_DIR%\lstm_model.pt" %PACKAGE_DIR%\
copy "%MODEL_DIR%\char2idx.pkl" %PACKAGE_DIR%\ 2>nul
copy "%MODEL_DIR%\training_history.pkl" %PACKAGE_DIR%\ 2>nul

REM Copy API server files
echo Copying API server scripts...
copy ml\api_server.py %PACKAGE_DIR%\api_server.py
copy ml\predict_lstm_api.py %PACKAGE_DIR%\predict_lstm_api.py
copy ml\requirements.txt %PACKAGE_DIR%\requirements.txt

REM Copy documentation
copy MODEL_RELEASE.md %PACKAGE_DIR%\README.md
copy .gitignore %PACKAGE_DIR%\.gitignore

REM Create version file
echo %VERSION% > %PACKAGE_DIR%\VERSION.txt
echo Created: March 13, 2026 >> %PACKAGE_DIR%\VERSION.txt
echo GPU: NVIDIA RTX 2050 >> %PACKAGE_DIR%\VERSION.txt

REM Create archive
echo Creating ZIP archive...
cd %PACKAGE_DIR%
cd ..
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('%PACKAGE_DIR%', 'lstm_model_release_v%VERSION%.zip')"

echo.
echo Package created: lstm_model_release_v%VERSION%.zip
echo.
echo Next steps:
echo 1. Go to: https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/new
echo 2. Tag version: v%VERSION%-model
echo 3. Title: LSTM Model Release v%VERSION%
echo 4. Upload file: lstm_model_release_v%VERSION%.zip
echo.
pause
