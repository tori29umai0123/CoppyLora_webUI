# CoppyLora_webUI

1枚のイラストから絵柄を学習するLoRA学習補助アプリ

## 動作環境
- **OS**: Windows 10 / 11
- **GPU**: NVIDIA GPU（CUDA 12.8対応、RTX 5090含む）

## ビルド設定（開発者向け）

### 必要環境
- **Python**: 3.11.x
- **CUDA**: 12.8
- **uv**: パッケージマネージャー
- **Git**

### セットアップ

1. `CoppyLora_webUI_install.ps1` をPowerShellで実行

```powershell
powershell -ExecutionPolicy Bypass -File CoppyLora_webUI_install.ps1
```

以下が自動的にセットアップされます：
- uv（未インストールの場合）
- Python 3.11 仮想環境（`.venv`）
- sd-scripts（コミット `8b5ce3e` 固定）
- PyTorch 2.10.0+cu128 / torchvision 0.25.0+cu128
- xformers 0.0.34（cu128）
- triton-windows / numpy 1.26.4
- gradio 4.3.0 / huggingface-hub 0.34.3
- bitsandbytes / wandb 0.17.3
- onnxruntime 1.17.1 / onnxruntime-gpu 1.17.1
- PyInstaller

2. セキュリティソフトの除外設定

ビルド・実行時にセキュリティソフトが干渉する場合があります。
Windows Defenderの場合：Windows セキュリティ > ウイルスと脅威の防止 > 設定の管理 > 除外 で以下を追加してください。
- `CoppyLora_webUI.exe`（プロセス）
- プロジェクトフォルダ（フォルダ）

### ビルド

`build.cmd` を実行するだけでビルドが完了します。

```
build.cmd
```

以下の処理が自動で行われます：
- CUDA 12.8 環境変数の設定
- 仮想環境の有効化
- 前回ビルドのクリーン（`models` フォルダは自動退避・復元）
- PyInstaller によるビルド
- プロジェクトファイル（caption, png, config.toml 等）のコピー
- ビルド成果物の検証

出力先: `dist\CoppyLora_webUI\`

### 開発時の仮想環境利用

`venv.cmd` を実行すると、仮想環境を有効化した状態でコマンドプロンプトが開きます。

```
venv.cmd
```
