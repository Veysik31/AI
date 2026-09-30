@echo off
echo ================================================================
echo RAM-YUM AI Stock Replenishment System - Installation
echo ================================================================
echo.

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)
echo.

echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [3/4] Installing Python dependencies (this may take 5-10 minutes)...
echo.
echo Installing packages one by one for better compatibility...
python -m pip install flask
python -m pip install flask-cors
python -m pip install mysql-connector-python
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install joblib
python -m pip install python-dateutil
python -m pip install pandas
echo.

echo [4/4] Creating directories...
if not exist "trained_models" mkdir trained_models
if not exist "data" mkdir data
echo.

echo ================================================================
echo Installation Complete!
echo ================================================================
echo.
echo To start the AI service, run:
echo     python app.py
echo.
echo Then access the dashboard at:
echo     http://localhost/RMS/pages/ai/dashboard.php
echo.
pause
