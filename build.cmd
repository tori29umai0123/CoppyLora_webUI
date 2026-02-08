@echo off
setlocal enabledelayedexpansion

echo ============================================
echo CoppyLora_webUI Build Script
echo ============================================
echo.

REM ---------------------------------------------------
REM 1. 前提条件チェック
REM ---------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv が見つかりません。先に CoppyLora_webUI_install.ps1 を実行してください。
    pause
    exit /b 1
)

if not exist "sd-scripts" (
    echo ERROR: sd-scripts が見つかりません。先に CoppyLora_webUI_install.ps1 を実行してください。
    pause
    exit /b 1
)

REM ---------------------------------------------------
REM 2. CUDA 12.8 環境変数設定
REM ---------------------------------------------------
echo Setting CUDA 12.8 environment...
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

REM ---------------------------------------------------
REM 3. 仮想環境を有効化
REM ---------------------------------------------------
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM ---------------------------------------------------
REM 4. 前回のビルド成果物を削除（modelsフォルダは退避）
REM ---------------------------------------------------
echo Cleaning previous build artifacts...
if exist "build" rmdir /s /q "build"

set "MODELS_SAVED=0"
if exist "dist\CoppyLora_webUI\models" (
    echo models フォルダを一時退避中...
    if exist "_models_backup" rmdir /s /q "_models_backup"
    move "dist\CoppyLora_webUI\models" "_models_backup"
    set "MODELS_SAVED=1"
)
if exist "dist" rmdir /s /q "dist"

REM ---------------------------------------------------
REM 5. PyInstaller 実行
REM ---------------------------------------------------
echo.
echo Running PyInstaller...
echo CUDA DLLを含むため、ビルドには時間がかかります。
echo.
pyinstaller CoppyLora_webUI.spec --noconfirm

if %errorlevel% neq 0 (
    echo.
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

REM ---------------------------------------------------
REM 6. post-build: exeレベルにプロジェクトファイルをコピー
REM ---------------------------------------------------
echo.
echo Copying project data files...

xcopy /E /I /Y "caption" "dist\CoppyLora_webUI\caption"
xcopy /E /I /Y "png" "dist\CoppyLora_webUI\png"
copy /Y "config.toml" "dist\CoppyLora_webUI\config.toml"
copy /Y "CoppyLora_webUI_ReadMe.txt" "dist\CoppyLora_webUI\CoppyLora_webUI_ReadMe.txt"
copy /Y "CoppyLora_webUI_DL.cmd" "dist\CoppyLora_webUI\CoppyLora_webUI_DL.cmd"

REM ---------------------------------------------------
REM 7. 退避したmodelsフォルダを復元
REM ---------------------------------------------------
if !MODELS_SAVED! equ 1 (
    if exist "_models_backup" (
        echo models フォルダを復元中...
        move "_models_backup" "dist\CoppyLora_webUI\models"
    )
)

REM ---------------------------------------------------
REM 8. ビルド検証
REM ---------------------------------------------------
echo.
echo Verifying build output...

set "FAIL=0"

if not exist "dist\CoppyLora_webUI\CoppyLora_webUI.exe" (
    echo MISSING: CoppyLora_webUI.exe
    set "FAIL=1"
)
if not exist "dist\CoppyLora_webUI\_internal\sd-scripts\sdxl_train_network.py" (
    echo MISSING: sd-scripts\sdxl_train_network.py
    set "FAIL=1"
)
if not exist "dist\CoppyLora_webUI\_internal\utils\tagger.py" (
    echo MISSING: utils\tagger.py
    set "FAIL=1"
)
if not exist "dist\CoppyLora_webUI\config.toml" (
    echo MISSING: config.toml
    set "FAIL=1"
)
if not exist "dist\CoppyLora_webUI\caption" (
    echo MISSING: caption directory
    set "FAIL=1"
)
if not exist "dist\CoppyLora_webUI\png" (
    echo MISSING: png directory
    set "FAIL=1"
)

REM xformersの存在確認（ビルドエラーの根本原因だったパッケージ）
dir /s /b "dist\CoppyLora_webUI\_internal\xformers" >nul 2>nul
if %errorlevel% neq 0 (
    echo MISSING: xformers package -- xformers ModuleNotFoundError が再発します
    set "FAIL=1"
)

if %FAIL% equ 1 (
    echo.
    echo BUILD VERIFICATION FAILED
    pause
    exit /b 1
)

echo.
echo ============================================
echo Build completed successfully!
echo Output: dist\CoppyLora_webUI\
echo ============================================
echo.
pause
