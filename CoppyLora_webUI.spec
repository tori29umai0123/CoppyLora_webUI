# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    collect_data_files,
    copy_metadata,
)

# =====================================================
# collect_all で各パッケージのdatas/binaries/hiddenimportsを収集
# =====================================================

# xformers: PyInstallerビルドで ModuleNotFoundError の原因だったパッケージ
# diffusers内部で条件付きインポートされるため、静的解析では検出されない
xformers_datas, xformers_binaries, xformers_hiddenimports = collect_all('xformers')

# torch / torchvision: CUDA runtime, cudnn, cublas等のDLLを含む
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')

# diffusers: xformersを条件付きインポートする本体
diffusers_datas, diffusers_binaries, diffusers_hiddenimports = collect_all('diffusers')

# transformers: diffusersとsd-scriptsが使用
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')

# accelerate: 学習オーケストレーション
accelerate_datas, accelerate_binaries, accelerate_hiddenimports = collect_all('accelerate')

# gradio: Web UI
gradio_datas, gradio_binaries, gradio_hiddenimports = collect_all('gradio')
gradio_client_datas, gradio_client_binaries, gradio_client_hiddenimports = collect_all('gradio_client')

# safetensors: モデル読み込み
safetensors_datas, safetensors_binaries, safetensors_hiddenimports = collect_all('safetensors')

# pytorch_lightning / lightning_fabric
pl_datas, pl_binaries, pl_hiddenimports = collect_all('pytorch_lightning')
lf_datas, lf_binaries, lf_hiddenimports = collect_all('lightning_fabric')

# cv2 (OpenCV): taggerで使用
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')

# bitsandbytes: AdamW8bitオプティマイザ
bnb_datas, bnb_binaries, bnb_hiddenimports = collect_all('bitsandbytes')

# onnxruntime: WD14 taggerで使用
ort_datas, ort_binaries, ort_hiddenimports = collect_all('onnxruntime')

# triton: xformersの最適化に使用 (Windows版)
triton_datas, triton_binaries, triton_hiddenimports = collect_all('triton')

# imagesize
imagesize_datas = collect_data_files('imagesize')

# transformersが起動時にimportlib.metadataでバージョンチェックするパッケージのメタデータ
regex_metadata = copy_metadata('regex')
filelock_metadata = copy_metadata('filelock')
numpy_metadata = copy_metadata('numpy')
requests_metadata = copy_metadata('requests')
tqdm_metadata = copy_metadata('tqdm')
pyyaml_metadata = copy_metadata('pyyaml')
packaging_metadata = copy_metadata('packaging')
safetensors_metadata = copy_metadata('safetensors')
tokenizers_metadata = copy_metadata('tokenizers')

# voluptuous: sd-scripts依存
voluptuous_hiddenimports = collect_submodules('voluptuous')

# =====================================================
# 全収集結果をマージ
# =====================================================
all_datas = (
    xformers_datas + torch_datas + torchvision_datas
    + diffusers_datas + transformers_datas + accelerate_datas
    + gradio_datas + gradio_client_datas + safetensors_datas
    + pl_datas + lf_datas + cv2_datas
    + bnb_datas + ort_datas + triton_datas + imagesize_datas
    + regex_metadata + filelock_metadata + numpy_metadata
    + requests_metadata + tqdm_metadata + pyyaml_metadata
    + packaging_metadata + safetensors_metadata + tokenizers_metadata
    + [
        # プロジェクト固有データ (_internal配下)
        ('sd-scripts', 'sd-scripts'),
        ('utils', 'utils'),
    ]
)

all_binaries = (
    xformers_binaries + torch_binaries + torchvision_binaries
    + diffusers_binaries + transformers_binaries + accelerate_binaries
    + gradio_binaries + gradio_client_binaries + safetensors_binaries
    + pl_binaries + lf_binaries + cv2_binaries
    + bnb_binaries + ort_binaries + triton_binaries
)

all_hiddenimports = list(set(
    xformers_hiddenimports + torch_hiddenimports + torchvision_hiddenimports
    + diffusers_hiddenimports + transformers_hiddenimports + accelerate_hiddenimports
    + gradio_hiddenimports + gradio_client_hiddenimports + safetensors_hiddenimports
    + pl_hiddenimports + lf_hiddenimports + cv2_hiddenimports
    + voluptuous_hiddenimports + bnb_hiddenimports + ort_hiddenimports + triton_hiddenimports
    + [
        # xformers: diffusers内部で条件付きインポートされるモジュール
        'xformers',
        'xformers.ops',
        'xformers.ops.memory_efficient_attention',
        'xformers.components',
        'xformers.components.attention',
        # torch CUDA
        'torch._C',
        'torch.cuda',
        'torch.backends.cudnn',
        'torch.utils.data',
        'torch.nn.functional',
        'torch.distributed',
        # diffusers attention (xformersインポートが発生する箇所)
        'diffusers.models.attention_processor',
        'diffusers.models.modeling_utils',
        'diffusers.pipelines.stable_diffusion',
        'diffusers.pipelines.stable_diffusion.safety_checker',
        # transformers
        'transformers.models.clip',
        'transformers.models.clip.modeling_clip',
        # toml (config.toml解析)
        'toml',
        # PIL
        'PIL',
        'PIL.Image',
        # numpy
        'numpy',
        # onnx / onnxruntime (tagger)
        'onnx',
        'onnxruntime',
        # gradio
        'gradio',
        'gradio.themes',
        'gradio_client',
        # safetensors
        'safetensors',
        'safetensors.torch',
        # accelerate
        'accelerate',
        'accelerate.utils',
        # lightning
        'pytorch_lightning',
        'lightning_fabric',
        # bitsandbytes
        'bitsandbytes',
        'bitsandbytes.optim',
        # imagesize
        'imagesize',
        # voluptuous (sd-scripts依存)
        'voluptuous',
        # wandb (sd-scriptsが任意でインポート)
        'wandb',
        # triton (xformers最適化)
        'triton',
    ]
))

# =====================================================
# Analysis
# =====================================================
a = Analysis(
    ['CoppyLora_webUI.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # サイズ削減のため不要なパッケージを除外
        'pytest',
        'tkinter',
        '_tkinter',
        'IPython',
        'jupyter',
        'notebook',
    ],
    noarchive=False,
    optimize=0,
    module_collection_mode={
        'gradio': 'py',
        'gradio_client': 'py',
        'xformers': 'py',
    },
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CoppyLora_webUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CoppyLora_webUI',
)
