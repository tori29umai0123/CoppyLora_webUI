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
git checkout "bfb352bc433326a77aca3124248331eb60c49e8c"

# 必要なパッケージをインストール
pip install --upgrade -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
pip install gradio==3.41.2
pip install pyinstaller

# ファイルを置換
$filePath = "library\lpw_stable_diffusion.py"
$oldLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker"
$newLine1 = "from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput"
$newLine2 = "from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker"

if ((Get-Content $filePath) -match $oldLine1) {
    (Get-Content $filePath) -replace $oldLine1, ($newLine1 + "`n" + $newLine2) | Set-Content $filePath
    Write-Output "File has been modified."
} else {
    Write-Output "No modification needed."
}

cd ..

Write-Output "Installation completed"
Read-Host -Prompt "Press Enter to exit"
