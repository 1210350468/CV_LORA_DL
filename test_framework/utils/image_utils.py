"""
图像处理工具
提供图像相关的处理函数
"""

import torch
from PIL import Image
import numpy as np
import os

def save_images(data, output_dir, task_id, logger):
    """保存图像"""
    logger.log_workflow(f"Saving images to {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for j in range(data.size(0)):
            for k in range(data.size(1)):
                # 反归一化
                img_array = denormalize_image(data[j][k])
                
                # 转换为PIL图像
                img = convert_to_pil(img_array)
                
                # 保存图像
                filename = f"{output_dir}/image_{task_id}_{j}_{k}.png"
                img.save(filename)
                
                logger.log_workflow(f"Saved image {filename}")
                
    except Exception as e:
        logger.log_error(f"Error saving images: {str(e)}")
        raise

def denormalize_image(img_tensor):
    """反归一化图像"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    
    return img_tensor

def convert_to_pil(img_array):
    """转换为PIL图像"""
    img_array = img_array.permute(1, 2, 0).numpy()
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array) 