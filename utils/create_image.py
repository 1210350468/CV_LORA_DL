#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint
import cv2

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import os
import json
import random
import shutil
from clip_similarity import CLIPSimilarityCalculator
prompts_file = '/root/autodl-tmp/all_starnight/Mytrain/utils/prompts.json'
images_dir = '/root/autodl-tmp/all_starnight/Mytrain/test/support_images/5_zkz'
output_dir = '/root/autodl-tmp/all_starnight/ComfyUI/output/'  # 替换为实际的 output 目录路径
support_images_dir = '/root/autodl-tmp/all_starnight/Mytrain/test/support_images/Lora_images'  # 替换为实际的 support_images 目录路径
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
def filter_images(image_path, support_images_dir, image_prefix, similarity_threshold=0.75):
    """
    使用CLIP计算图片相似度，使用平均相似度作为筛选标准
    Args:
        image_path: 需要检查的图片路径
        support_images_dir: support图片目录
        image_prefix: 图片类别前缀
        similarity_threshold: 相似度阈值
    Returns:
        bool: True表示保留，False表示过滤掉
    """
    try:
        calculator = CLIPSimilarityCalculator()
        
        # 获取support目录下所有同前缀的图片路径
        support_images = []
        for file in os.listdir(support_images_dir):
            if file.startswith(image_prefix) and is_image_file(file):
                support_images.append(os.path.join(support_images_dir, file))
        
        if not support_images:
            print(f"No support images found for prefix {image_prefix}, keeping the image")
            return True
        
        # 计算与所有support图片的相似度
        similarities = []
        for support_img_path in support_images:
            similarity = calculator.compute_similarity(image_path, support_img_path)
            similarities.append(similarity)
        
        # 计算平均相似度
        avg_similarity = sum(similarities) / len(similarities)
        
        # 打印相似度信息
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Average similarity: {avg_similarity:.4f}")
        print(f"Min similarity: {min(similarities):.4f}")
        print(f"Max similarity: {max(similarities):.4f}")
        print(f"Keep image: {avg_similarity >= similarity_threshold}")
        
        # 使用平均相似度作为判断标准
        return avg_similarity >= similarity_threshold
        
    except Exception as e:
        print(f"Error in filter_images: {str(e)}")
        return False
    
def generate_single_image(ws, client_id, prompt, category, support_images_dir, max_attempts=999):
    """
    生成单张图片并检查相似度，如果不满足要求则重新生成
    
    Args:
        ws: websocket连接
        client_id: 客户端ID
        prompt: 生成图片的prompt
        image_path: 原始图片路径
        category: 图片类别
        support_images_dir: support图片目录
        max_attempts: 最大尝试次数
    
    Returns:
        bool: 是否成功生成满足条件的图片
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"\nAttempt {attempt}/{max_attempts} for category {category}")
        
        # 生成新的随机种子
        seed = random.randint(0, 2**64-1)
        prompt["3"]["inputs"]["seed"] = seed
        
        # 生成图片
        get_images(ws, prompt, client_id)
        generated_files = [f for f in os.listdir(output_dir) if f'LoraAux' in f and f.startswith(category)]
        
        if not generated_files:
            print("No image generated, retrying...")
            continue
            
        generated_path = os.path.join(output_dir, generated_files[-1])
        
        # 检查相似度
        if filter_images(generated_path, support_images_dir, category):
            print(f"Successfully generated image for {category} on attempt {attempt}")
            return True
        else:
            print(f"Generated image similarity too low, removing and retrying...")
            os.remove(generated_path)
    
    print(f"Failed to generate satisfactory image for {category} after {max_attempts} attempts")
    return False

def generate_images_from_prompts(prompts_file, images_dir, num_images_per_category=5):
    """
    为每个类别生成指定数量的图片，确保每张图片都满足相似度要求
    """
    # 初始化连接
    ws, client_id = initialize_connection()
    
    # 加载提示词
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # 获取现有图片的类别信息
    categorized_images = {}
    for image_file in os.listdir(images_dir):
        if not is_image_file(image_file):
            continue
        
        image_prefix = os.path.splitext(image_file)[0].split('_')[0]
        if image_prefix not in categorized_images:
            categorized_images[image_prefix] = []
        categorized_images[image_prefix].append(image_file)
    
    # 对每个类别的图片进行排序
    for category, images in categorized_images.items():
        images.sort()
    
    # 生成图片
    successful_generations = {}
    for category, images in categorized_images.items():
        if category not in prompts:
            continue
            
        successful_generations[category] = 0
        prompt_data = prompts[category]
        
        print(f"\nProcessing category: {category}")
        print(f"Target: {num_images_per_category} images")
        
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
            prompt_text = get_workflow()
            prompt = json.loads(json.dumps(prompt_text))
            prompt["6"]["inputs"]["text"] = positive_prompt
            prompt["7"]["inputs"]["text"] = prompts["negative"]
            prompt["9"]["inputs"]["filename_prefix"] = f"{category}_LoraAux"
            prompt["10"]["inputs"]["image"] = image_path
            
            # 生成图片直到满足要求
            if generate_single_image(ws, client_id, prompt,category, images_dir):
                successful_generations[category] += 1
            
        print(f"\nCategory {category} complete:")
        print(f"Successfully generated {successful_generations[category]}/{num_images_per_category} images")
    
    return successful_generations

def move_loraaux_images(output_dir, support_images_dir):
    """
    移动所有生成的图片到目标目录
    """
    if not os.path.exists(support_images_dir):
        os.makedirs(support_images_dir)

    moved_count = 0
    for filename in os.listdir(output_dir):
        if 'LoraAux' in filename.split('_'):
            src_path = os.path.join(output_dir, filename)
            dest_path = os.path.join(support_images_dir, filename)
            shutil.move(src_path, dest_path)
            moved_count += 1
            print(f"Moved: {src_path} -> {dest_path}")
    
    print(f"\nMoved {moved_count} images to {support_images_dir}")

def get_workflow():
    prompt_text ={
    "3": {
      "inputs": {
        "seed": 189808209495139,
        "steps": 30,
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
        "width": 512,
        "height": 512,
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

generate_images_from_prompts(prompts_file, images_dir)

move_loraaux_images(output_dir, support_images_dir)





