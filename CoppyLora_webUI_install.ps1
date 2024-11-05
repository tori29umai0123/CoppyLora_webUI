Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

if (!(Test-Path -Path "venv")) {
    Write-Output "Creating Python virtual environment..."
    python -m venv venv
}
.\venv\Scripts\Activate.ps1

python.exe -m pip install --upgrade pip

if (!(Test-Path -Path "sd-scripts")) {
    git clone https://github.com/kohya-ss/sd-scripts.git
}
cd sd-scripts
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
pip install -U bitsandbytes
pip install wandb==0.17.3
pip install gradio==4.44.1
pip install pyinstaller
pip install onnx==1.15.0 onnxruntime==1.17.1 onnxruntime-gpu==1.17.1

$filePath1 = "library\lpw_stable_diffusion.py"
$filePath2 = "library\sdxl_lpw_stable_diffusion.py"

$oldLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker"
$newLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput"
$newLine2 = "from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker"

if ((Get-Content $filePath1) -match $oldLine1) {
    (Get-Content $filePath1) -replace $oldLine1, ($newLine1 + "`n" + $newLine2) | Set-Content $filePath1
    Write-Output "File $filePath1 has been modified."
} else {
    Write-Output "No modification needed for $filePath1."
}

if ((Get-Content $filePath2) -match $oldLine1) {
    (Get-Content $filePath2) -replace $oldLine1, ($newLine1 + "`n" + $newLine2) | Set-Content $filePath2
    Write-Output "File $filePath2 has been modified."
} else {
    Write-Output "No modification needed for $filePath2."
}
cd ..

Write-Output "Installation completed"
Read-Host -Prompt "Press Enter to exit"
