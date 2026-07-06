@echo off
title AI Assistant - Gemini Vision
color 0A

echo.
echo  ====================================
echo   AI ASSISTANT - GEMINI VISION TOOL
echo  ====================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python chua duoc cai dat!
    echo Tai Python tai: https://python.org
    pause
    exit /b 1
)

:: Install dependencies
echo [1/2] Dang cai dat dependencies...
pip install PyQt5 Pillow google-generativeai --quiet --upgrade

echo [2/2] Dang khoi dong AI Assistant...
echo.
python main.py

pause
