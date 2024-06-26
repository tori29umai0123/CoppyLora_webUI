# 現在のスクリプトのディレクトリに移動する
Set-Location $PSScriptRoot

# pipのバージョンチェックを無効化する
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

# Pythonの仮想環境を作成する
if (!(Test-Path -Path "venv")) {
    Write-Output "Creating Python virtual environment..."
    python -m venv venv
}
.\venv\Scripts\Activate.ps1

# pipをアップグレードする
python.exe -m pip install --upgrade pip

# リポジトリをクローンして特定のコミットにチェックアウトする
if (!(Test-Path -Path "sd-scripts")) {
    git clone https://github.com/kohya-ss/sd-scripts.git
}
cd sd-scripts
git checkout "25f961bc779bc79aef440813e3e8e92244ac5739"

# 必要なパッケージをインストール
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
pip install -U bitsandbytes
pip install wandb==0.17.3
pip install gradio==4.37.1
pip install pyinstaller

# 定義されたファイルパス
$filePath1 = "library\lpw_stable_diffusion.py"
$filePath2 = "library\sdxl_lpw_stable_diffusion.py"

# 置き換える古い行と新しい行
$oldLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker"
$newLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput"
$newLine2 = "from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker"

# ファイル1の処理
if ((Get-Content $filePath1) -match $oldLine1) {
    (Get-Content $filePath1) -replace $oldLine1, ($newLine1 + "`n" + $newLine2) | Set-Content $filePath1
    Write-Output "File $filePath1 has been modified."
} else {
    Write-Output "No modification needed for $filePath1."
}

# ファイル2の処理
if ((Get-Content $filePath2) -match $oldLine1) {
    (Get-Content $filePath2) -replace $oldLine1, ($newLine1 + "`n" + $newLine2) | Set-Content $filePath2
    Write-Output "File $filePath2 has been modified."
} else {
    Write-Output "No modification needed for $filePath2."
}
cd ..

Write-Output "Installation completed"
Read-Host -Prompt "Press Enter to exit"
