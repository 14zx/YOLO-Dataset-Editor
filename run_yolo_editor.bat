@echo off
title YOLO Dataset Editor - Full Setup
echo ==================================================
echo   YOLO Dataset Editor - Full Setup
echo ==================================================
echo.

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found! Download from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

:: Check Node.js
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js not found! Download from https://nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js found:
node --version
echo.

:: Python venv
set "VENV_DIR=%USERPROFILE%\yolo-editor-venv"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/3] Creating Python venv: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo.
    echo       Installing packages...
    call "%VENV_DIR%\Scripts\activate.bat"
    pip install --upgrade pip >nul 2>&1
    pip install ultralytics flask flask-cors
    echo.
    echo       Done!
    echo.
) else (
    echo [1/3] Python venv exists: %VENV_DIR%
    call "%VENV_DIR%\Scripts\activate.bat"
    :: Make sure flask is installed
    pip show flask >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo       Installing flask...
        pip install flask flask-cors
    )
    echo.
)

set "PATH=%VENV_DIR%\Scripts;%PATH%"

if not exist "%~dp0setup.js" (
    echo [ERROR] setup.js not found!
    pause
    exit /b 1
)
if not exist "%~dp0yolo-dataset-editor.jsx" (
    echo [ERROR] yolo-dataset-editor.jsx not found!
    pause
    exit /b 1
)
if not exist "%~dp0yolo_server.py" (
    echo [ERROR] yolo_server.py not found!
    pause
    exit /b 1
)

echo [2/3] Setting up React project...
echo [3/3] Starting servers...
echo.
node "%~dp0setup.js"

pause
