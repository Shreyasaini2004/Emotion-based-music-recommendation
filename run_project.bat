@echo off
echo 🎵 Emotion-Based Music Recommender 🎵
echo =====================================
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo Choose an option:
echo 1. Collect emotion data
echo 2. Train model
echo 3. Run main application
echo.
set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" (
    echo Starting data collection...
    python data_collection.py
) else if "%choice%"=="2" (
    echo Starting model training...
    python train_model.py
) else if "%choice%"=="3" (
    echo Starting main application...
    streamlit run main_app.py
) else (
    echo Invalid choice. Please run the script again.
)

pause