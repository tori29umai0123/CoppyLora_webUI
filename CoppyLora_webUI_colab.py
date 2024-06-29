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
import subprocess


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

spec_cache_latents = importlib.util.spec_from_file_location("cache_latents", os.path.join(tools_path, 'cache_latents.py'))
cache_latents = importlib.util.module_from_spec(spec_cache_latents)
spec_cache_latents.loader.exec_module(cache_latents)


config_file = os.path.join(path, "config.toml")
models_dir = os.path.join(path, "models")
train_data_dir = os.path.join(path, "train_data")
image_4000_dir = os.path.join(train_data_dir, "4000")
image_1_dir = os.path.join(train_data_dir, "1")
base_image_path = os.path.join(models_dir, "base_c_1024.png")
base_c_lora = os.path.join(models_dir, "copi-ki-base-c.safetensors")
base_b_lora = os.path.join(models_dir, "copi-ki-base-b.safetensors")
base_cnl_lora = os.path.join(models_dir, "copi-ki-base-cnl.safetensors")
base_bnl_lora = os.path.join(models_dir, "copi-ki-base-bnl.safetensors")
caption_dir = os.path.join(path, "caption")
SDXL_model = os.path.join(models_dir, "animagine-xl-3.1.safetensors")

accelerate_config = os.path.join(path, "accelerate_config.yaml")
import os
from accelerate.utils import write_basic_config

if not os.path.exists(accelerate_config):
    write_basic_config(save_location=accelerate_config)

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
        caption_size_txt = os.path.join(image_1_dir, f"{size}.txt")
        with open(caption_txt, "r") as f:
            with open(caption_size_txt, "w") as f2:
                f2.write(f.read())

def train(input_image_path, lora_name, mode_inputs):
    if os.path.exists(image_1_dir):
        shutil.rmtree(image_1_dir)
    os.makedirs(image_1_dir)

    if os.path.exists(image_4000_dir):
        shutil.rmtree(image_4000_dir)

    output_dir = os.path.join(path, "output")
    
    input_image = Image.open(input_image_path)
    base_lora = setup_base_lora(mode_inputs)

    for size in [1024, 768, 512]:
        resize_image = input_image.resize((size, size))
        resize_image.save(os.path.join(image_1_dir, f"{size}.png"))

    #学習前にcache_latentsを作る
    args_dict = {
        "pretrained_model_name_or_path": SDXL_model,
        "train_data_dir": train_data_dir,
        "output_dir": models_dir,
        "output_name": "copi-ki-kari",
        "max_train_steps": 1000,
        "xformers": True,
        "gradient_checkpointing": True,
        "persistent_data_loader_workers": True,
        "max_data_loader_n_workers": 12,
        "enable_bucket": True,
        "resolution": "1024,1024",
        "train_batch_size": 2,
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "bucket_no_upscale": True,
        "min_bucket_reso": 64,
        "max_bucket_reso": 1024,
        "caption_extension": ".txt",
        "seed": 42,
        "no_half_vae": True,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "sdxl": True,
        "skip_existing": True,
        "console_log_simple": True,
        "lowram": True
    }

    parser = cache_latents.setup_parser()
    args = parser.parse_args()
    args = cache_latents.train_util.read_config_from_file(args, parser)
    args2 = argparse.Namespace(**args_dict)
    for key, value in vars(args2).items():
        setattr(args, key, value)
    cache_latents.cache_to_disk(args)
    #つぎの学習の為にフォルダ名を元に戻す
    os.rename(image_1_dir, image_4000_dir)
    sdxl_train_network = os.path.join(sd_scripts_dir, 'sdxl_train_network.py')
    command1 = [
        "accelerate", "launch", "--config_file", accelerate_config, sdxl_train_network,
        "--pretrained_model_name_or_path", SDXL_model,
        "--train_data_dir", train_data_dir,
        "--output_dir", models_dir,
        "--output_name", "copi-ki-kari",
        "--max_train_steps", "1000",
        "--network_module", "networks.lora",
        "--xformers",
        "--gradient_checkpointing",
        "--persistent_data_loader_workers",
        "--max_data_loader_n_workers", "12",
        "--enable_bucket",
        "--save_model_as", "safetensors",
        "--lr_scheduler_num_cycles", "4",
        "--learning_rate", "1e-4",
        "--resolution", "1024,1024",
        "--train_batch_size", "2",
        "--network_dim", "16",
        "--network_alpha", "16",
        "--optimizer_type", "AdamW8bit",
        "--mixed_precision", "fp16",
        "--save_precision", "fp16",
        "--lr_scheduler", "constant",
        "--bucket_no_upscale",
        "--min_bucket_reso", "64",
        "--max_bucket_reso", "1024",
        "--caption_extension", ".txt",
        "--seed", "42",
        "--network_train_unet_only",
        "--no_half_vae",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--fp8_base"
    ]
    subprocess.run(command1, check=True, cwd=sd_scripts_dir)   

    kari_lora = os.path.join(models_dir, "copi-ki-kari.safetensors")
    output_dir = os.path.join(path, "output")
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
    demo.launch(share=True, server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
