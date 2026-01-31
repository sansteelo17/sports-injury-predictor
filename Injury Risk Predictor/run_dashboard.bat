@echo off
REM Football Injury Risk Predictor - Dashboard Launcher (Windows)
REM This script provides an easy way to start the Streamlit dashboard

echo ========================================
echo Football Injury Risk Predictor
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)
echo Python detected
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Check if dependencies are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo Dependencies installed
    echo.
) else (
    echo Dependencies already installed
    echo.
)

REM Check if app.py exists
if not exist "app.py" (
    echo Error: app.py not found
    echo Make sure you're running this script from the project root directory
    pause
    exit /b 1
)

REM Start the dashboard
echo ========================================
echo Starting Streamlit Dashboard...
echo ========================================
echo.
echo Dashboard will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit
streamlit run app.py
