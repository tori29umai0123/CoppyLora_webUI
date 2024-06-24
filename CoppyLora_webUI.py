import gradio as gr
import torch
import subprocess
import os
import sys
from PIL import Image
import importlib.util
import argparse
import socket
import webbrowser
import threading

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

spec_sdxl_train_network = importlib.util.spec_from_file_location("sdxl_train_network", os.path.join(sd_scripts_dir, 'sdxl_train_network.py'))
sdxl_train_network = importlib.util.module_from_spec(spec_sdxl_train_network)
spec_sdxl_train_network.loader.exec_module(sdxl_train_network)


data_dir = os.path.join(path, "data")
train_data_dir = os.path.join(path, "train")
SDXL_model = os.path.join(data_dir, "animagine-xl-3.1.safetensors")
base_image_path = os.path.join(data_dir, "base_c_1024.png")
base_c_lora = os.path.join(data_dir, "copi-ki-base-c.safetensors")
base_b_lora = os.path.join(data_dir, "copi-ki-base-b.safetensors")
base_cnl_lora = os.path.join(data_dir, "copi-ki-base-cnl.safetensors")
base_bnl_lora = os.path.join(data_dir, "copi-ki-base-bnl.safetensors")

def find_free_port():
    """空いているポートを見つけて返す関数"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))  # ランダムな空いているポートを割り当てる
    port = sock.getsockname()[1]
    sock.close()
    return port

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
    input_image = Image.open(input_image_path)
    if mode_inputs == "Lineart":
        base_lora = base_c_lora
        image_dir = os.path.join(train_data_dir, "lineart/4000")
        train_dir = os.path.join(train_data_dir, "lineart")

    if mode_inputs == "Grayscale":
        base_lora = base_cnl_lora
        image_dir = os.path.join(train_data_dir, "grayscale/4000")
        train_dir = os.path.join(train_data_dir, "grayscale")

    if mode_inputs == "Grayscale_noline":
        base_lora = base_c_lora
        image_dir = os.path.join(train_data_dir, "grayscale_noline/4000")
        train_dir = os.path.join(train_data_dir, "grayscale_noline")

    if mode_inputs == "Color":
        base_lora = base_bnl_lora
        image_dir = os.path.join(train_data_dir, "grayscale/4000")
        train_dir = os.path.join(train_data_dir, "grayscale")

    if mode_inputs == "Color_noline":
        base_lora = base_b_lora
        image_dir = os.path.join(train_data_dir, "color_noline/4000")
        train_dir = os.path.join(train_data_dir, "color_noline")

    for size in [1024, 768, 512]:
        resize_image = input_image.resize((size, size))
        resize_image.save(os.path.join(image_dir, f"{size}.png"))

    args_dict = {
        "pretrained_model_name_or_path": SDXL_model,
        "train_data_dir": train_dir,
        "output_dir": data_dir,
        "output_name": "copi-ki-kari",
        "max_train_steps": 1000,
        "network_module": "networks.lora",
        "xformers": True,
        "gradient_checkpointing": True,
        "persistent_data_loader_workers": True,
        "max_data_loader_n_workers": 12,
        "enable_bucket": True,
        "save_model_as": "safetensors",
        "lr_scheduler_num_cycles": 4,
        "learning_rate": 1e-4,
        "resolution": "1024,1024",
        "train_batch_size": 2,
        "network_dim": 16,
        "network_alpha": 16,
        "optimizer_type": "AdamW8bit",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "lr_scheduler": "constant",
        "bucket_no_upscale": True,
        "min_bucket_reso": 64,
        "max_bucket_reso": 1024,
        "caption_extension": ".txt",
        "seed": 42,
        "network_train_unet_only": True,
        "no_half_vae": True,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
    }

    Low_VRAM = check_cuda()
    if Low_VRAM:
        args_dict.append({"fp8_base": True})   
 
    parser = sdxl_train_network.setup_parser()
    args = parser.parse_args()
    sdxl_train_network.train_util.verify_command_line_training_args(args)
    args = sdxl_train_network.train_util.read_config_from_file(args, parser)
    args2 = argparse.Namespace(**args_dict)
    for key, value in vars(args2).items():
        setattr(args, key, value)
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    trainer.train(args)

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
    demo.launch(share=False, server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
