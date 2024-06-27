@echo off
setlocal

set "base_dir=%~dp0models"

if not exist "%base_dir%" mkdir "%base_dir%"

rem モデルファイルのダウンロード
call :downloadFile "%base_dir%" "animagine-xl-3.1.safetensors" "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors"
call :downloadFile "%base_dir%" "copi-ki-base-c.safetensors" "https://huggingface.co/tori29umai/mylora/resolve/main/copi-ki-base-c.safetensors"
call :downloadFile "%base_dir%" "copi-ki-base-b.safetensors" "https://huggingface.co/tori29umai/mylora/resolve/main/copi-ki-base-b.safetensors"
call :downloadFile "%base_dir%" "copi-ki-base-cnl.safetensors" "https://huggingface.co/tori29umai/mylora/resolve/main/copi-ki-base-cnl.safetensors"
call :downloadFile "%base_dir%" "copi-ki-base-bnl.safetensors" "https://huggingface.co/tori29umai/mylora/resolve/main/copi-ki-base-bnl.safetensors"
call :downloadFile "%base_dir%" "base_c_1024.png" "https://huggingface.co/tori29umai/mylora/resolve/main/base_c_1024.png?download=true"

goto :eof

:downloadFile
set "file_path=%~1\%~2"
if exist "%file_path%" (
    echo %~2 already exists.
    goto :eof
)

curl -o "%file_path%" -L %~3
if %errorlevel% equ 0 (
    echo Downloaded %~2
) else (
    echo Failed to download %~2
)
goto :eof