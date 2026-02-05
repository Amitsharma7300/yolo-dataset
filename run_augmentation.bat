@echo off
REM Quick Start Script for Machinery Augmentation Pipeline
REM ========================================================

echo ====================================================
echo Heavy Augmentation Pipeline for Construction Machinery
echo ====================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Step 1: Installing dependencies...
pip install -r requirements.txt
echo.

echo Step 2: Choose your workflow:
echo.
echo [1] Run annotation tool first (recommended for best results)
echo [2] Run augmentation directly (uses auto-generated bounding boxes)
echo [3] Exit
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting annotation tool...
    echo Draw bounding boxes around machinery parts.
    echo Press 'n' for next image, 'q' to quit when done.
    echo.
    python annotate_bboxes.py --input_dir .
    echo.
    echo Annotation complete! Now running augmentation...
    python heavy_augmentation.py --input_dir . --output_dir augmented_dataset --num_augmentations 50 --extreme
) else if "%choice%"=="2" (
    echo.
    echo Running augmentation with auto-generated bounding boxes...
    python heavy_augmentation.py --input_dir . --output_dir augmented_dataset --num_augmentations 50 --extreme
) else (
    echo Exiting...
    exit /b 0
)

echo.
echo ====================================================
echo Augmentation complete!
echo Output saved to: augmented_dataset\
echo ====================================================
pause
