
#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import os
import json
import random
import shutil
def initialize_connection():
  server_address = "127.0.0.1:8188"
  client_id = str(uuid.uuid4())
  ws = websocket.WebSocket()
  ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
  return ws, client_id
def queue_prompt(prompt,client_id,server_address = "127.0.0.1:8188"):

    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    print(server_address, req, data)
    return json.loads(urllib.request.urlopen(req).read())



def get_image(filename, subfolder, folder_type,server_address = "127.0.0.1:8188"):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()




def get_history(prompt_id,server_address = "127.0.0.1:8188"):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())




def get_images(ws, prompt,client_id):

    print(client_id)
    prompt_id = queue_prompt(prompt,client_id)['prompt_id']
    print(prompt_id)
    output_images = {}
    while True:
        out = ws.recv()
        print(out)
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                # print(data)
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data
  
    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
       
        for node_id in history['outputs']:
            
            node_output = history['outputs'][node_id]
            # print(node_output)
            if 'images' in node_output:
                
                images_output = []
                for image in node_output['images']:
                   
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    # print(image_data)
                    images_output.append(image_data)
            output_images[node_id] = images_output
    # print(output_images)
    return output_images






# 从外部JSON文件加载提示词
def is_image_file(file_name):
    # 定义常见图片文件后缀
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    # 获取文件扩展名并转为小写
    ext = os.path.splitext(file_name)[1].lower()
    return ext in image_extensions

def generate_images_from_prompts(prompts_file, images_dir, num_images_per_category=5):
    # 初始化 WebSocket 连接
    ws, client_id = initialize_connection()
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # 创建一个字典来存储按类别分类的图片
    categorized_images = {}
    #support_lora
    # 遍历图片目录，将图片按前缀分类
    for image_file in os.listdir(images_dir):
        if not is_image_file(image_file):
            continue  # 如果文件不是图片，则跳过
        
        image_prefix = os.path.splitext(image_file)[0].split('_')[0]
        if image_prefix not in categorized_images:
            categorized_images[image_prefix] = []
        categorized_images[image_prefix].append(image_file)
    
    # 对每个类别中的图片进行排序
    for category, images in categorized_images.items():
        images.sort()
    print(categorized_images)
    # input('...')
    # 按类别顺序和类别中的图片顺序生成图片
    for category, images in categorized_images.items():
        for i in range(min(num_images_per_category, len(images))):
            image_file = images[i]
            image_path = os.path.join(images_dir, image_file)
            
            if category in prompts:
                prompt_data = prompts[category]
            else:
                continue  # 如果类别不在提示词字典中，跳过
            variable_prompt= random.choice(prompt_data["variable"])
            # 组合正面提示词
            positive_prompt = f"{prompt_data['fixed']}, {variable_prompt}"
            
            # 随机生成一个种子
            seed = random.randint(0, 2**64-1)
            prompt_text = get_workflow()
            # 载入模板并更新提示词和种子
            json_string = json.dumps(prompt_text)
            prompt = json.loads(json_string)
            prompt["6"]["inputs"]["text"] = positive_prompt
            prompt["7"]["inputs"]["text"] = prompts["negative"]
            prompt["3"]["inputs"]["seed"] = seed
            prompt["9"]["inputs"]["filename_prefix"] = category + '_LoraAux'
            prompt["10"]["inputs"]["image"] = image_path  # 使用获取到的图片路径
            
            # 提交生成请求
            get_images(ws, prompt, client_id)
def get_workflow():
    prompt_text ={
    "3": {
      "inputs": {
        "seed": 189808209495139,
        "steps": 25,
        "cfg": 7,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1,
        "model": [
          "17",
          0
        ],
        "positive": [
          "6",
          0
        ],
        "negative": [
          "7",
          0
        ],
        "latent_image": [
          "13",
          0
        ]
      },
      "class_type": "KSampler",
      "_meta": {
        "title": "K采样器"
      }
    },
    "4": {
      "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
      },
      "class_type": "CheckpointLoaderSimple",
      "_meta": {
        "title": "Checkpoint加载器(简易)"
      }
    },
    "6": {
      "inputs": {
        "text": "'satellite image',\"River\"",
        "clip": [
          "17",
          1
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP文本编码器"
      }
    },
    "7": {
      "inputs": {
        "text": "",
        "clip": [
          "17",
          1
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP文本编码器"
      }
    },
    "8": {
      "inputs": {
        "samples": [
          "3",
          0
        ],
        "vae": [
          "20",
          0
        ]
      },
      "class_type": "VAEDecode",
      "_meta": {
        "title": "VAE解码"
      }
    },
    "9": {
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": [
          "16",
          0
        ]
      },
      "class_type": "SaveImage",
      "_meta": {
        "title": "保存图像"
      }
    },
    "10": {
      "inputs": {
        "image": "",
        "upload": "image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "加载图像"
      }
    },
    "11": {
      "inputs": {
        "pixels": [
          "10",
          0
        ],
        "vae": [
          "20",
          0
        ]
      },
      "class_type": "VAEEncode",
      "_meta": {
        "title": "VAE编码"
      }
    },
    "13": {
      "inputs": {
        "upscale_method": "nearest-exact",
        "width": 256,
        "height": 256,
        "crop": "disabled",
        "samples": [
          "11",
          0
        ]
      },
      "class_type": "LatentUpscale",
      "_meta": {
        "title": "Latent缩放"
      }
    },
    "16": {
      "inputs": {
        "upscale_method": "nearest-exact",
        "width": 256,
        "height": 256,
        "crop": "disabled",
        "image": [
          "8",
          0
        ]
      },
      "class_type": "ImageScale",
      "_meta": {
        "title": "图像缩放"
      }
    },
    "17": {
      "inputs": {
        "lora_name": "support_lora.safetensors",
        "strength_model": 1,
        "strength_clip": 1,
        "model": [
          "4",
          0
        ],
        "clip": [
          "4",
          1
        ]
      },
      "class_type": "LoraLoader",
      "_meta": {
        "title": "LoRA加载器"
      }
    },
    "20": {
      "inputs": {
        "vae_name": "sdxl_vae.safetensors"
      },
      "class_type": "VAELoader",
      "_meta": {
        "title": "VAE加载器"
      }
    }
    }
    return prompt_text
# 调用函数
def move_loraaux_images(output_dir, support_images_dir):
    # 确保 support_images 目录存在
    if not os.path.exists(support_images_dir):
        os.makedirs(support_images_dir)

    # 遍历 output 目录中的文件
    for filename in os.listdir(output_dir):
        # 使用 split 方法判断文件名中是否包含 '_LoraAux_'
        if 'LoraAux' in filename.split('_'):
            # 构建源文件路径和目标文件路径
            src_path = os.path.join(output_dir, filename)
            dest_path = os.path.join(support_images_dir, filename)
            
            # 移动文件
            shutil.move(src_path, dest_path)
            print(f"Moved: {src_path} -> {dest_path}")


generate_images_from_prompts('/root/autodl-tmp/all_starnight/Mytrain/utils/prompts.json', '/root/autodl-tmp/all_starnight/Mytrain/test/support_images/5_zkz')
output_dir = '/root/autodl-tmp/all_starnight/ComfyUI/output/'  # 替换为实际的 output 目录路径
support_images_dir = '/root/autodl-tmp/all_starnight/Mytrain/test/support_images/Lora_images'  # 替换为实际的 support_images 目录路径
move_loraaux_images(output_dir, support_images_dir)





