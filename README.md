# CoppyLora_webUI


# ビルド設定（開発者向け）
このプロジェクトのビルドには以下の環境が必要です：
- **Python**: 3.11.x
- **CUDA**: 12.1

これらのバージョンに依存するライブラリやモジュールがありますので、互換性のある環境でビルドしてください。
sd-scriptsライブラリを使用しているため、ビルド時にはバージョンを合わせた上で以下の手順が必要です。
①CoppyLora_webUI.py.ps1を実行してインストール<br>
②セキュリティーソフトの設定で、フォルダと実行ファイル名を除外リストに追加する。<br>
例：Windows Defenderの場合、Windows セキュリティ→ウイルスと脅威の防止→ウイルスと脅威の防止の設定→設定の管理→除外<br>
CoppyLora_webUI.py.exe(プロセス)<br>
C:\CoppyLora_webUI.py（フォルダ）<br>
のように指定する。<br>

## 実行ファイル生成
venv.cmdを実行し、以下のコマンドを入力
```
pyinstaller CoppyLora_webUI.spec

xcopy /E /I /Y venv\Lib\site-packages\gradio dist\CoppyLora_webUI\_internal\gradio
xcopy /E /I /Y venv\Lib\site-packages\gradio_client dist\CoppyLora_webUI\_internal\gradio_client
xcopy /E /I /Y venv\Lib\site-packages\xformers dist\CoppyLora_webUI\_internal\xformers
xcopy /E /I /Y venv\Lib\site-packages\accelerate dist\CoppyLora_webUI\_internal\accelerate
xcopy /E /I /Y venv\Lib\site-packages\pytorch_lightning dist\CoppyLora_webUI\_internal\pytorch_lightning
xcopy /E /I /Y venv\Lib\site-packages\lightning_fabric dist\CoppyLora_webUI\_internal\lightning_fabric
xcopy /E /I /Y venv\Lib\site-packages\diffusers dist\CoppyLora_webUI\_internal\diffusers
xcopy /E /I /Y venv\Lib\site-packages\imagesize dist\CoppyLora_webUI\_internal\imagesize
xcopy /E /I /Y venv\Lib\site-packages\cv2 dist\CoppyLora_webUI\_internal\cv2
xcopy /E /I /Y venv\Lib\site-packages\voluptuous dist\CoppyLora_webUI\_internal\voluptuous
xcopy /E /I /Y caption dist\CoppyLora_webUI\caption
xcopy /E /I /Y png dist\CoppyLora_webUI\png
xcopy /E /I /Y utils dist\CoppyLora_webUI\_internal\utils

copy CoppyLora_webUI_ReadMe.txt dist\CoppyLora_webUI\CoppyLora_webUI_ReadMe.txt
copy CoppyLora_webUI_DL.cmd dist\CoppyLora_webUI\CoppyLora_webUI_DL.cmd
copy config.toml dist\CoppyLora_webUI\config.toml
```
