import os
import shutil
# ログでエラーが出るので、念のため環境変数を設定
os.environ['TERM'] = 'dumb'
import gradio as gr
import torch
import sys
from PIL import Image
import importlib.util
import argparse
import socket
import webbrowser
import threading
import socket
import toml

# ビルドしているかしていないかでパスを変更
if getattr(sys, 'frozen', False):
    path = os.path.dirname(sys.executable)
    sd_scripts_dir = os.path.join(path, "_internal", 'sd-scripts')
    networks_path = os.path.join(sd_scripts_dir, 'networks')
    library_path = os.path.join(sd_scripts_dir, 'library')
    tools_path = os.path.join(sd_scripts_dir, 'tools')
else:
    path = os.path.dirname(os.path.abspath(__file__))
    sd_scripts_dir = os.path.join(path, 'sd-scripts')
    networks_path = os.path.join(sd_scripts_dir, 'networks')
    library_path = os.path.join(sd_scripts_dir, 'library')
    tools_path = os.path.join(sd_scripts_dir, 'tools')

# パスをシステムパスに追加
sys.path.append(sd_scripts_dir)
sys.path.append(networks_path)
sys.path.append(library_path)
sys.path.append(tools_path)

# モジュールのパスを直接指定してインポート
spec_merge = importlib.util.spec_from_file_location("merge", os.path.join(networks_path, 'sdxl_merge_lora.py'))
merge = importlib.util.module_from_spec(spec_merge)
spec_merge.loader.exec_module(merge)

spec_resize = importlib.util.spec_from_file_location("resize", os.path.join(networks_path, 'resize_lora.py'))
resize = importlib.util.module_from_spec(spec_resize)
spec_resize.loader.exec_module(resize)

config_file = os.path.join(path, "config.toml")
models_dir = os.path.join(path, "models")
train_data_dir = os.path.join(path, "train_data")
image_dir = os.path.join(train_data_dir, "4000")
base_image_path = os.path.join(models_dir, "base_c_1024.png")
base_c_lora = os.path.join(models_dir, "copi-ki-base-c.safetensors")
base_b_lora = os.path.join(models_dir, "copi-ki-base-b.safetensors")
base_cnl_lora = os.path.join(models_dir, "copi-ki-base-cnl.safetensors")
base_bnl_lora = os.path.join(models_dir, "copi-ki-base-bnl.safetensors")
caption_dir = os.path.join(path, "caption")

def find_free_port(start_port=7860):
    """指定したポートから開始して空いているポートを見つけて返す関数"""
    for port in range(start_port, 65535):  # 65535はポート番号の最大値
        try:
            # ソケットを作成して指定のポートにバインドを試みる
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port  # バインドに成功したらそのポート番号を返す
        except OSError:
            # バインドに失敗した場合（ポートが使用中の場合）、次のポートを試す
            continue
    raise RuntimeError("No free ports available.")  # 空いているポートが見つからなかった場合

def setup_base_lora(mode_inputs):
    if mode_inputs in ["Lineart", "Grayscale_noline"]:
        base_lora = base_c_lora
    elif mode_inputs == "Grayscale":
        base_lora = base_cnl_lora
    elif mode_inputs == "Color":
        base_lora = base_bnl_lora
    elif mode_inputs == "Color_noline":
        base_lora = base_b_lora

    return base_lora


def setup_caption(mode_inputs):
    if mode_inputs in ["Lineart", "Grayscale_noline"]:
        caption_txt = os.path.join(caption_dir, "color_g.txt")
    else :
        caption_txt = os.path.join(caption_dir, "grayscale_g.txt")

    #caption_txtをtmp_dirに名前をそれぞれ1024.txt, 768.txt, 512.txtに変更してコピー
    for size in [1024, 768, 512]:
        caption_size_txt = os.path.join(image_dir, f"{size}.txt")
        with open(caption_txt, "r") as f:
            with open(caption_size_txt, "w") as f2:
                f2.write(f.read())


def check_cuda():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_memory <= 15:
            print("Low VRAM detected, using fp8_base")
            Low_VRAM = True           
        else:
            print("High VRAM detected, using fp16_base")
            Low_VRAM = False
    return Low_VRAM

def train(input_image_path, lora_name, mode_inputs):
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)
    output_dir = os.path.join(path, "output")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    input_image = Image.open(input_image_path)
    base_lora = setup_base_lora(mode_inputs)

    for size in [1024, 768, 512]:
        resize_image = input_image.resize((size, size))
        resize_image.save(os.path.join(image_dir, f"{size}.png"))

    spec_sdxl_train_network = importlib.util.spec_from_file_location("sdxl_train_network", os.path.join(sd_scripts_dir, 'sdxl_train_network.py'))
    sdxl_train_network = importlib.util.module_from_spec(spec_sdxl_train_network)
    spec_sdxl_train_network.loader.exec_module(sdxl_train_network)

    # config.tomlファイルの読み込み
    with open(config_file, 'r') as f:
        config = toml.load(f)

    # 読み込んだ設定をargs_dictに代入
    args_dict = {
        "pretrained_model_name_or_path": config["pretrained_model_name_or_path"],
        "train_data_dir": config["train_data_dir"],
        "output_dir": config["output_dir"],
        "output_name": config["output_name"],
        "max_train_steps": config["max_train_steps"],
        "network_module": config["network_module"],
        "xformers": config["xformers"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "persistent_data_loader_workers": config["persistent_data_loader_workers"],
        "max_data_loader_n_workers": config["max_data_loader_n_workers"],
        "enable_bucket": config["enable_bucket"],
        "save_model_as": config["save_model_as"],
        "lr_scheduler_num_cycles": config["lr_scheduler_num_cycles"],
        "learning_rate": config["learning_rate"],
        "resolution": config["resolution"],
        "train_batch_size": config["train_batch_size"],
        "network_dim": config["network_dim"],
        "network_alpha": config["network_alpha"],
        "optimizer_type": config["optimizer_type"],
        "mixed_precision": config["mixed_precision"],
        "save_precision": config["save_precision"],
        "lr_scheduler": config["lr_scheduler"],
        "bucket_no_upscale": config["bucket_no_upscale"],
        "min_bucket_reso": config["min_bucket_reso"],
        "max_bucket_reso": config["max_bucket_reso"],
        "caption_extension": config["caption_extension"],
        "seed": config["seed"],
        "network_train_unet_only": config["network_train_unet_only"],
        "no_half_vae": config["no_half_vae"],
        "cache_latents": config["cache_latents"],
        "cache_latents_to_disk": config["cache_latents_to_disk"],
        "cache_text_encoder_outputs": config["cache_text_encoder_outputs"],
        "cache_text_encoder_outputs_to_disk": config["cache_text_encoder_outputs_to_disk"],
    }

    Low_VRAM = check_cuda()
    if Low_VRAM:
        args_dict["fp8_base"] = True
 
    parser = sdxl_train_network.setup_parser()
    args = parser.parse_args()
    sdxl_train_network.train_util.verify_command_line_training_args(args)
    args = sdxl_train_network.train_util.read_config_from_file(args, parser)
    args2 = argparse.Namespace(**args_dict)
    for key, value in vars(args2).items():
        setattr(args, key, value)
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    trainer.train(args)

    kari_lora = os.path.join(models_dir, "copi-ki-kari.safetensors")
    merge_lora = os.path.join(models_dir, "merge_lora.safetensors")
    train_lora = os.path.join(output_dir, f"{lora_name}.safetensors")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args_dict = {
        "save_precision": "bf16",
        "precision": "float",
        "sd_model": None,
        "save_to": merge_lora,
        "models": [kari_lora, base_lora],
        "no_metadata": False,
        "ratios": [1.41, -1.41],
        "concat": True,
        "shuffle": True,
    }
    args = argparse.Namespace(**args_dict)   
    merge.merge(args)

    args_dict = {
        "save_precision": "bf16",
        "new_rank":16,
        "new_conv_rank": 16,
        "save_to": train_lora,
        "model": merge_lora,
        "device": "cuda",
        "verbose": "store_true",
        "dynamic_param": None,
        "dynamic_method": None,
    }
    args = argparse.Namespace(**args_dict)  
    resize.resize(args)
    return train_lora

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                base_img = gr.Image(value=base_image_path, label="Base Image")
                input_image_path = gr.Image(label="Input Image", type='filepath')
                lora_name = gr.Textbox(label="LoRa Name", value="mylora")
                mode_inputs = gr.Dropdown(label="Mode", choices=["Lineart","Grayscale","Grayscale_noline","Color","Color_noline"], value="Lineart")
                train_button = gr.Button("Train")
            with gr.Column():
                output_file = gr.File(label="Download Output File")

        train_button.click(
            fn=train,
            inputs=[input_image_path, lora_name, mode_inputs],
            outputs=output_file
        )
    demo.queue()
    # 空いているポートを取得
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"  # 空いているポートを使用する

    # ブラウザでURLを開く
    threading.Thread(target=lambda: webbrowser.open_new(url)).start()
    is_colab = 'COLAB_GPU' in os.environ
    share_setting = True if is_colab else False
    demo.launch(share=share_setting, server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
