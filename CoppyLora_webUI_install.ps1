Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

# -------------------------------------------------------
# 1. uvがインストールされていなければインストール
# -------------------------------------------------------
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Output "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
}

# -------------------------------------------------------
# 2. Python 3.11の仮想環境を作成
# -------------------------------------------------------
if (!(Test-Path -Path ".venv")) {
    Write-Output "Creating Python 3.11 virtual environment with uv..."
    uv venv --python 3.11 .venv
}

# -------------------------------------------------------
# 3. sd-scriptsをクローン
# -------------------------------------------------------
if (!(Test-Path -Path "sd-scripts")) {
    git clone https://github.com/kohya-ss/sd-scripts.git
    git -C sd-scripts checkout 8b5ce3e641f0cf0775546922353105cb2d3a6895
}

# -------------------------------------------------------
# 4. PyTorch (cu128 / RTX 5090対応)
# -------------------------------------------------------
Write-Output "Installing PyTorch 2.10.0+cu128..."
uv pip install --python .venv torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# -------------------------------------------------------
# 5. sd-scriptsの依存パッケージ
#    requirements.txtの最終行に "-e ." があるため、sd-scriptsディレクトリで実行
# -------------------------------------------------------
Write-Output "Installing sd-scripts requirements..."
$venvAbsPath = (Resolve-Path ".venv").Path
cd sd-scripts
uv pip install --python $venvAbsPath -r requirements.txt
cd ..

# -------------------------------------------------------
# 6. xformers (cu128対応)
# -------------------------------------------------------
Write-Output "Installing xformers 0.0.34..."
uv pip install --python .venv xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu128

# -------------------------------------------------------
# 7. triton (Windows版) / numpy固定
# -------------------------------------------------------
Write-Output "Installing triton-windows and numpy..."
uv pip install --python .venv triton-windows
uv pip install --python .venv numpy==1.26.4

# -------------------------------------------------------
# 8. 追加パッケージ
# -------------------------------------------------------
Write-Output "Installing additional packages..."
uv pip install --python .venv bitsandbytes
uv pip install --python .venv wandb==0.17.3
uv pip install --python .venv gradio==4.3.0
uv pip install --python .venv huggingface-hub==0.34.3
uv pip install --python .venv pyinstaller
uv pip install --python .venv onnx==1.15.0 onnxruntime==1.17.1 onnxruntime-gpu==1.17.1
uv pip install --python .venv toml

# -------------------------------------------------------
# 9. diffusersのインポートパッチ (sd-scripts内)
# -------------------------------------------------------
$filePath1 = "sd-scripts\library\lpw_stable_diffusion.py"
$filePath2 = "sd-scripts\library\sdxl_lpw_stable_diffusion.py"

$oldLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker"
$newLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput"
$newLine2 = "from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker"

foreach ($filePath in @($filePath1, $filePath2)) {
    if (Test-Path $filePath) {
        if ((Get-Content $filePath) -match [regex]::Escape($oldLine1)) {
            (Get-Content $filePath) -replace [regex]::Escape($oldLine1), ($newLine1 + "`n" + $newLine2) | Set-Content $filePath
            Write-Output "File $filePath has been modified."
        } else {
            Write-Output "No modification needed for $filePath."
        }
    }
}

Write-Output ""
Write-Output "Installation completed!"
Read-Host -Prompt "Press Enter to exit"
