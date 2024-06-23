# CoppyLora_webUI

# インストール
```
python -m venv venv
.\venv\Scripts\activate
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
pip install gradio==3.41.2
pip install pyinstaller
```
# ビルド設定（開発者向け）
python 3.11.xで開発されています。
sd-scriptsライブラリを使用しているため、ビルド時にはバージョンを合わせた上で以下の手順が必要です。

①AI_Assistant_install.ps1を実行してインストール<br>
②セキュリティーソフトの設定で、フォルダと実行ファイル名を除外リストに追加する。<br>
例：Windows Defenderの場合、Windows セキュリティ→ウイルスと脅威の防止→ウイルスと脅威の防止の設定→設定の管理→除外<br>
AI_Assistant.exe(プロセス)<br>
C:\AI_Assistant（フォルダ）<br>
のように指定する。<br>

## 実行ファイル生成
venv.cmdを実行し、以下のコマンドを入力
```
pyinstaller "CoppyLora_webUI.py" ^
--clean ^
--copy-metadata rich ^
--add-data "sd-scripts;.sd-scripts" ^
--add-binary "venv\Scripts\accelerate.exe;.accelerate.exe"

xcopy /E /I /Y venv\Lib\site-packages\accelerate dist\CoppyLora_webUI\_internal\accelerate
xcopy /E /I /Y venv\Lib\site-packages\xformers dist\CoppyLora_webUI\_internal\xformers
xcopy /E /I /Y venv\Lib\site-packages\pytorch_lightning dist\CoppyLora_webUI\_internal\pytorch_lightning
xcopy /E /I /Y venv\Lib\site-packages\lightning_fabric dist\CoppyLora_webUI\_internal\lightning_fabric
xcopy /E /I /Y venv\Lib\site-packages\gradio dist\CoppyLora_webUI\_internal\gradio
xcopy /E /I /Y venv\Lib\site-packages\gradio_client dist\CoppyLora_webUI\_internal\gradio_client
xcopy /E /I /Y venv\Lib\site-packages\diffusers dist\CoppyLora_webUI\_internal\diffusers
xcopy /E /I /Y venv\Lib\site-packages\imagesize dist\CoppyLora_webUI\_internal\imagesize
xcopy /E /I /Y venv\Lib\site-packages\cv2 dist\CoppyLora_webUI\_internal\cv2
xcopy /E /I /Y train dist\CoppyLora_webUI\train
xcopy /E /I /Y sd-scripts dist\CoppyLora_webUI\sd-scripts
copy CoppyLora_webUI_ReadMe.txt dist\CoppyLora_webUI\CoppyLora_webUI_ReadMe.txt
copy CoppyLora_webUI_DL.cmd dist\CoppyLora_webUI\CoppyLora_webUI_DL.cmd
copy venv\Scripts\accelerate.exe dist\CoppyLora_webUI\accelerate.exe

```
