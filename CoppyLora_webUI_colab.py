import os
#colab環境に合わせる
os.environ['TERM'] = 'dumb'
import gradio as gr
import torch
import subprocess
import sys
from PIL import Image
import importlib.util
import argparse
import socket
import webbrowser
import threading
import socket

def get_appropriate_file_path():
    if getattr(sys, 'frozen', False):
        path = os.path.dirname(sys.executable)
        return path
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        return path
    
path = get_appropriate_file_path()

sd_scripts_dir = os.path.join(path, 'sd-scripts')
networks_path = os.path.join(sd_scripts_dir, 'networks')
library_path = os.path.join(sd_scripts_dir, 'library')

# パスをシステムパスに追加
sys.path.append(sd_scripts_dir)
sys.path.append(networks_path)
sys.path.append(library_path)

# モジュールのパスを直接指定してインポート
spec_merge = importlib.util.spec_from_file_location("merge", os.path.join(networks_path, 'sdxl_merge_lora.py'))
merge = importlib.util.module_from_spec(spec_merge)
spec_merge.loader.exec_module(merge)

spec_resize = importlib.util.spec_from_file_location("resize", os.path.join(networks_path, 'resize_lora.py'))
resize = importlib.util.module_from_spec(spec_resize)
spec_resize.loader.exec_module(resize)

data_dir = os.path.join(path, "data")
train_data_dir = os.path.join(path, "train")
SDXL_model = os.path.join(data_dir, "animagine-xl-3.1.safetensors")
base_image_path = os.path.join(data_dir, "base_c_1024.png")
base_c_lora = os.path.join(data_dir, "copi-ki-base-c.safetensors")
base_b_lora = os.path.join(data_dir, "copi-ki-base-b.safetensors")
base_cnl_lora = os.path.join(data_dir, "copi-ki-base-cnl.safetensors")
base_bnl_lora = os.path.join(data_dir, "copi-ki-base-bnl.safetensors")

accelerate_config = "/content/CoppyLora_Train/accelerate_config.yaml"
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

def setup_paths(mode_inputs, train_data_dir):
    if mode_inputs in ["Lineart", "Grayscale_noline"]:
        base_lora = base_c_lora
    elif mode_inputs == "Grayscale":
        base_lora = base_cnl_lora
    elif mode_inputs == "Color":
        base_lora = base_bnl_lora
    elif mode_inputs == "Color_noline":
        base_lora = base_b_lora

    dir_type = mode_inputs.lower().replace('_', '')
    old_image_dir = os.path.join(train_data_dir, f"{dir_type}/1")
    new_image_dir = os.path.join(train_data_dir, f"{dir_type}/4000")
    train_dir = os.path.join(train_data_dir, dir_type)

    return base_lora, old_image_dir, new_image_dir, train_dir

def train(input_image_path, lora_name, mode_inputs):
    input_image = Image.open(input_image_path)
    base_lora, old_image_dir, new_image_dir, train_dir = setup_paths(mode_inputs, train_data_dir)

    #もしold_image_dirがなかったら、new_image_dirをold_image_dirにリネームする
    if not os.path.exists(old_image_dir):
        os.rename(new_image_dir, old_image_dir)

    for size in [1024, 768, 512]:
        resize_image = input_image.resize((size, size))
        resize_image.save(os.path.join(old_image_dir, f"{size}.png"))

    cache_latents = os.path.join(sd_scripts_dir, 'tools/cache_latents.py')

    #学習前にcache_latentsを作る
    command = [
        "accelerate", "launch", "--config_file", accelerate_config, cache_latents,
        "--pretrained_model_name_or_path", SDXL_model,
        "--train_data_dir", train_dir,
        "--output_dir", data_dir,
        "--output_name", "copi-ki-kari",
        "--max_train_steps", "1000",
        "--xformers",
        "--gradient_checkpointing",
        "--persistent_data_loader_workers",
        "--max_data_loader_n_workers", "12",
        "--enable_bucket",
        "--resolution", "1024,1024",
        "--train_batch_size", "2",
        "--mixed_precision", "fp16",
        "--save_precision", "fp16",
        "--bucket_no_upscale",
        "--min_bucket_reso", "64",
        "--max_bucket_reso", "1024",
        "--caption_extension", ".txt",
        "--seed", "42",
        "--no_half_vae",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--sdxl",
        "--skip_existing",
        "--console_log_simple",
        "--lowram"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training step: {e}")
        return  # 学習中のエラーが発生した場合もここで処理を終了
         
    #つぎの学習の為にフォルダ名を元に戻す
    os.rename(old_image_dir, new_image_dir)
    sdxl_train_network = os.path.join(sd_scripts_dir, 'sdxl_train_network.py')
    command = [
        "accelerate", "launch", "--config_file", accelerate_config, sdxl_train_network,
        "--pretrained_model_name_or_path", SDXL_model,
        "--train_data_dir", train_dir,
        "--output_dir", data_dir,
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
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training step: {e}")
        return  # 学習中のエラーが発生した場合もここで処理を終了

    os.rename(new_image_dir, old_image_dir)

    kari_lora = os.path.join(data_dir, "copi-ki-kari.safetensors")
    output_dir = os.path.join(path, "output")
    merge_lora = os.path.join(data_dir, "merge_lora.safetensors")
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
