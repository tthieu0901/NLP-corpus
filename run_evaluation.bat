@echo off
REM Quick Model Evaluation Script
REM This script runs the model evaluation on the trained Chinese MLM

echo ========================================
echo   Chinese MLM Model Evaluation
echo ========================================
echo.

echo Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.7+ and add to PATH.
    pause
    exit /b 1
)

echo.
echo Checking if model exists...
if not exist "chinese_ming_history_mlm" (
    echo Error: Trained model not found in 'chinese_ming_history_mlm' directory.
    echo Please run build_predict_word_model.py first to train the model.
    pause
    exit /b 1
)

echo.
echo Running quick evaluation...
python quick_eval_model.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   Evaluation completed successfully!
    echo ========================================
    echo.
    echo Results files:
    echo - quick_evaluation_report.txt  (Summary report)
    echo - quick_evaluation_results.json (Detailed results)
    echo.
    echo To run comprehensive evaluation, use:
    echo python eval_predict_word_model.py
) else (
    echo.
    echo ========================================
    echo   Evaluation failed!
    echo ========================================
    echo Please check the error messages above.
)

echo.
pause
