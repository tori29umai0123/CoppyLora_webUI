@echo off
rem Activate the virtual environment
call venv\Scripts\activate

rem Set CUDA to version 12.1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

cmd /k