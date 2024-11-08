@echo off
setlocal enabledelayedexpansion

REM Set the base path for models and png directories
set "dpath=%~dp0models"
set "pngpath=%~dp0png"
echo Base directory for models set to: %dpath%
echo Base directory for png set to: %pngpath%

goto :main

:download_and_verify
set "FILE_PATH=%~1"
set "DOWNLOAD_URL=%~2"
set "EXPECTED_HASH=%~3"
set "MAX_ATTEMPTS=3"

for /L %%i in (1,1,%MAX_ATTEMPTS%) do (
    echo Attempt %%i of %MAX_ATTEMPTS%
    curl -L "!DOWNLOAD_URL!" -o "!FILE_PATH!"
    
    REM Calculate SHA-256 hash
    for /f "skip=1 tokens=* delims=" %%# in ('certutil -hashfile "!FILE_PATH!" SHA256') do (
        set "ACTUAL_HASH=%%#"
        goto :hash_calculated
    )
    :hash_calculated
    
    if "!ACTUAL_HASH!"=="!EXPECTED_HASH!" (
        echo Hash verification successful.
        exit /b 0
    ) else (
        echo Hash mismatch. Retrying...
        if %%i equ %MAX_ATTEMPTS% (
            echo Warning: Failed to download file with matching hash after %MAX_ATTEMPTS% attempts.
            exit /b 1
        )
    )
)
exit /b

:verify_hash
set "FILE_PATH=%~1"
set "EXPECTED_HASH=%~2"

for /f "skip=1 tokens=* delims=" %%# in ('certutil -hashfile "%FILE_PATH%" SHA256') do (
    set "ACTUAL_HASH=%%#"
    goto :hash_calculated_verify
)
:hash_calculated_verify

if "%ACTUAL_HASH%"=="%EXPECTED_HASH%" (
    echo Hash verification successful for %FILE_PATH%
    exit /b 0
) else (
    echo Hash mismatch for %FILE_PATH%
    echo Expected: %EXPECTED_HASH%
    echo Actual:   %ACTUAL_HASH%
    exit /b 1
)

:download_files_custom
if "%~1"=="" (
    echo No arguments provided to download_files_custom
    exit /b 1
)
echo Downloading files to "%~1" from "%~2" with custom path "%~4"
set "MODEL_DIR=%dpath%\%~1"
set "MODEL_ID=%~2"
set "FILES=%~3"
set "CUSTOM_PATH=%~4"
echo MODEL_DIR: !MODEL_DIR!
echo MODEL_ID: !MODEL_ID!
echo FILES: !FILES!
echo CUSTOM_PATH: !CUSTOM_PATH!

if not exist "!MODEL_DIR!" (
    echo Creating directory !MODEL_DIR!
    mkdir "!MODEL_DIR!"
)

for %%f in (%FILES%) do (
    set "FILE_PATH=!MODEL_DIR!\%%f"
    set "EXPECTED_HASH=!%~1_%%~nf_hash!"
    set "RETRY_COUNT=0"
    :retry_download_custom
    if not exist "!FILE_PATH!" (
        echo Downloading %%f...
        curl -L "https://huggingface.co/!MODEL_ID!/resolve/main/!CUSTOM_PATH!/%%f" -o "!FILE_PATH!"
        if !errorlevel! neq 0 (
            echo Error downloading %%f
        ) else (
            echo Downloaded %%f
            call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
            if !errorlevel! neq 0 (
                echo Hash verification failed for %%f
                set /a RETRY_COUNT+=1
                if !RETRY_COUNT! lss 3 (
                    echo Retry !RETRY_COUNT!/3
                    del "!FILE_PATH!"
                    goto :retry_download_custom
                ) else (
                    echo Hash verification failed after 3 retries. Deleting %%f
                    del "!FILE_PATH!"
                )
            )
        )
    ) else (
        echo %%f already exists. Verifying hash...
        call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
        if !errorlevel! neq 0 (
            echo Hash verification failed for existing file %%f
            del "!FILE_PATH!"
            set "RETRY_COUNT=0"
            goto :retry_download_custom
        )
    )
)
exit /b 0

:download_files_default
if "%~1"=="" (
    echo No arguments provided to download_files_default
    exit /b 1
)

REM Set directories based on argument
if "%~1"=="png" (
    set "MODEL_DIR=%pngpath%"
) else (
    set "MODEL_DIR=%dpath%\%~1"
)
set "MODEL_ID=%~2"
set "FILES=%~3"

echo MODEL_DIR: !MODEL_DIR!
echo MODEL_ID: !MODEL_ID!
echo FILES: !FILES!

if not exist "!MODEL_DIR!" (
    echo Creating directory !MODEL_DIR!
    mkdir "!MODEL_DIR!"
)

for %%f in (%FILES%) do (
    set "FILE_PATH=!MODEL_DIR!\%%f"
    set "EXPECTED_HASH=!%~1_%%~nf_hash!"
    set "RETRY_COUNT=0"
    :retry_download_default
    if not exist "!FILE_PATH!" (
        echo Downloading %%f...
        curl -L "https://huggingface.co/!MODEL_ID!/resolve/main/%%f" -o "!FILE_PATH!"
        if !errorlevel! neq 0 (
            echo Error downloading %%f
        ) else (
            echo Downloaded %%f
            call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
            if !errorlevel! neq 0 (
                echo Hash verification failed for %%f
                set /a RETRY_COUNT+=1
                if !RETRY_COUNT! lss 3 (
                    echo Retry !RETRY_COUNT!/3
                    del "!FILE_PATH!"
                    goto :retry_download_default
                ) else (
                    echo Hash verification failed after 3 retries. Deleting %%f
                    del "!FILE_PATH!"
                )
            )
        )
    ) else (
        echo %%f already exists. Verifying hash...
        call :verify_hash "!FILE_PATH!" "!EXPECTED_HASH!"
        if !errorlevel! neq 0 (
            echo Hash verification failed for existing file %%f
            del "!FILE_PATH!"
            set "RETRY_COUNT=0"
            goto :retry_download_default
        )
    )
)
exit /b 0

:main
echo Starting main execution

REM Define hashes
set "SDXL_animagine-xl-3.1_hash=e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0"
set "tagger_config_hash=ddcdd28facc40ee8d0ef4b16ee3e7c70e4d7b156aff7b0f2ccc180e617eda795"
set "tagger_model_hash=e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1"
set "tagger_selected_tags_hash=298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217"
set "tagger_sw_jax_cv_config_hash=4dda7ac5591de07f7444ca30f2f89971a21769f1db6279f92ca996d371b761c9"
set "LoRa_copi-ki-base-boy_cl_am31_hash=5aa481749352901b790a0128500d2b83ac63a4bb51eb3278e7da0d6976c6087d"
set "LoRa_copi-ki-base-boy_ncl_am31_hash=2b112eefd7203827606670df0a5d5f4fe61317a712180456ecc544d8b815964a"
set "LoRa_copi-ki-base-boy_ncnl_am31_hash=61c965a0c1e45262282511e6c1d40c2a2b226616bad9ea4e77c504951d4074c3"
set "LoRa_copi-ki-base-boy_cnl_am31_hash=1e96f710153fd33255526727f9241235617651c9e1df844885612182a2c50675"
set "LoRa_copi-ki-base-girl_cl_am31_hash=8979e0cee623891d15980ca1513cd6b5d97684129977dcad694524a6271bd0d1"
set "LoRa_copi-ki-base-girl_ncl_am31_hash=e14961aed6102b17b920dea89c7a30fbc48910b66ce833f775806f34ed581f68"
set "LoRa_copi-ki-base-girl_ncnl_am31_hash=f7d3d0f2bc9896751865cabf05de1d383b1d86d6b643a805868e9372bbb590d4"
set "LoRa_copi-ki-base-girl_cnl_am31_hash=f3925e4c51c1f2cb2dd339849713e2502808741f08e42246eb2b6e5f01f0c4ce"

echo Downloading Stable-diffusion model:
call :download_files_default "SDXL" "cagliostrolab/animagine-xl-3.1" "animagine-xl-3.1.safetensors"

echo Downloading Tagger model:
call :download_files_default "tagger" "SmilingWolf/wd-swinv2-tagger-v3" "config.json,model.onnx,selected_tags.csv,sw_jax_cv_config.json"

echo Downloading Lora models:
call :download_files_default "LoRa" "tori29umai/mylora_V2" "copi-ki-base-boy_cl_am31.safetensors,copi-ki-base-boy_ncl_am31.safetensors,copi-ki-base-boy_ncnl_am31.safetensors,copi-ki-base-boy_cnl_am31.safetensors,copi-ki-base-girl_cl_am31.safetensors,copi-ki-base-girl_ncl_am31.safetensors,copi-ki-base-girl_ncnl_am31.safetensors,copi-ki-base-girl_cnl_am31.safetensors,copi-ki-base-female_p_am31.safetensors,copi-ki-base-male_p_am31.safetensors"


echo Script execution completed
echo Press Enter to close the script...
pause > nul
exit /b
endlocal
