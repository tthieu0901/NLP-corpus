@echo off
REM Batch script to run MLM training in conda environment

echo ================================================
echo Ming History Masked Language Model Training
echo ================================================

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

echo Current conda environment: %CONDA_DEFAULT_ENV%

REM Activate the data-formulator environment
echo Activating data-formulator environment...
call conda activate data-formulator
if %errorlevel% neq 0 (
    echo Warning: Failed to activate data-formulator environment.
    echo Creating new environment...
    call conda create -n data-formulator python=3.9 -y
    call conda activate data-formulator
)

echo.
echo Environment activated: %CONDA_DEFAULT_ENV%

REM Install dependencies
echo Installing dependencies...
python setup_environment.py

REM Run data inspection (optional)
echo.
echo Do you want to inspect the training data first? (y/n)
set /p choice=
if /i "%choice%"=="y" (
    echo Running data inspection...
    python inspect_data.py
    echo.
)

REM Run training
echo Do you want to start training now? (y/n)
set /p choice=
if /i "%choice%"=="y" (
    echo Starting MLM training...
    echo This may take several hours...
    python build_predict_word_model.py
    
    echo.
    echo Training completed! Testing the model...
    python test_model.py
) else (
    echo Training skipped. To run training later, use:
    echo   python build_predict_word_model.py
)

echo.
echo Script completed!
pause
