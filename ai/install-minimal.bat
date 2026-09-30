@echo off
echo ================================================================
echo RAM-YUM AI - MINIMAL Installation (Without Pandas)
echo ================================================================
echo.
echo This installs core dependencies only.
echo Pandas will be installed separately if needed.
echo.

echo [1/3] Checking Python...
python --version
echo.

echo [2/3] Installing core packages...
python -m pip install --upgrade pip
python -m pip install flask flask-cors mysql-connector-python numpy scikit-learn joblib python-dateutil
echo.

echo [3/3] Creating directories...
if not exist "trained_models" mkdir trained_models
if not exist "data" mkdir data
echo.

echo ================================================================
echo Core Installation Complete!
echo ================================================================
echo.
echo Note: Pandas will be installed automatically when first used,
echo or you can install it manually:
echo     python -m pip install pandas
echo.
echo To start the AI service:
echo     python app.py
echo.
pause
