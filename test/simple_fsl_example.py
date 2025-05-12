import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import utils
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入FSLDataLoader
from dataloader.FSLDataLoader import FSLTaskLoader, get_combined_support

def save_tensor_as_image(tensor, save_path):
    """将tensor保存为图像文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为PIL图像并保存
    # 因为图像已经标准化，我们需要反标准化
    img = tensor.clone().detach().cpu()
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    # 确保值在[0, 1]范围内
    img = torch.clamp(img, 0, 1)
    # 转换为PIL图像
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img)
    # 保存
    img.save(save_path)

def save_grid_image(tensors, nrow, save_path):
    """将多个tensor创建为网格图像并保存"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建网格
    grid = utils.make_grid(tensors, nrow=nrow, padding=2, normalize=True)
    # 转换为numpy并保存
    grid = grid.permute(1, 2, 0).cpu().numpy()
    # 确保值在[0, 1]范围内
    grid = np.clip(grid, 0, 1)
    # 转换为PIL图像
    grid = (grid * 255).astype(np.uint8)
    img = Image.fromarray(grid)
    # 保存
    img.save(save_path)

def simple_example():
    """简单的FSLTaskLoader使用示例"""
    print("=== 简单的FSLTaskLoader使用示例 ===")
    
    # 创建保存图像的目录
    save_dir = os.path.join(os.path.dirname(__file__), "saved_task_images")
    os.makedirs(save_dir, exist_ok=True)
    print(f"图像将保存到: {save_dir}")
    
    # 1. 创建任务加载器
    task_loader = FSLTaskLoader(
        dataset_path="./task_images",
        n_way=5,          # 类别数
        n_support=5,      # 每类支持集样本数
        n_query=15        # 每类查询集样本数
    )
    
    print(f"总任务数: {len(task_loader)}")
    
    # 2. 直接获取某个任务的数据并保存图像
    task_indices = range(len(task_loader)) # 获取所有任务
    
    for task_idx in task_indices:
        support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info = task_loader.get_task(task_idx)
        task_name = task_info["task_name"]
        
        print(f"\n处理任务 {task_idx}: {task_name}")
        print(f"支持集形状: {support_images.shape}")        # [5, 5, 3, 224, 224]
        print(f"LoRA图像形状: {lora_images.shape}")        # [5, 5, 3, 224, 224]
        print(f"查询集形状: {query_images.shape}")         # [5, 15, 3, 224, 224]
        
        # 创建当前任务的保存目录
        task_save_dir = os.path.join(save_dir, f"task_{task_idx}_{task_name}")
        os.makedirs(task_save_dir, exist_ok=True)
        
        # 保存每个类别的第一张图片
        for class_idx in range(support_images.shape[0]):
            # 支持集
            save_tensor_as_image(
                support_images[class_idx, 0], 
                os.path.join(task_save_dir, f"support_class{class_idx}.png")
            )
            
            # LoRA集
            save_tensor_as_image(
                lora_images[class_idx, 0], 
                os.path.join(task_save_dir, f"lora_class{class_idx}.png")
            )
            
            # 查询集
            save_tensor_as_image(
                query_images[class_idx, 0], 
                os.path.join(task_save_dir, f"query_class{class_idx}.png")
            )
        
        # 保存类别网格图像
        for class_idx in range(support_images.shape[0]):
            # 支持集网格
            save_grid_image(
                support_images[class_idx], 
                nrow=support_images.shape[1],
                save_path=os.path.join(task_save_dir, f"support_class{class_idx}_grid.png")
            )
            
            # LoRA网格
            save_grid_image(
                lora_images[class_idx], 
                nrow=lora_images.shape[1],
                save_path=os.path.join(task_save_dir, f"lora_class{class_idx}_grid.png")
            )
            
            # 查询集网格(保存前10张)
            save_grid_image(
                query_images[class_idx], 
                nrow=5,
                save_path=os.path.join(task_save_dir, f"query_class{class_idx}_grid.png")
            )
        
        # 合并支持集和LoRA图像
        combined_support = get_combined_support(support_images, lora_images)
        
        # 保存合并后的网格
        for class_idx in range(combined_support.shape[0]):
            save_grid_image(
                combined_support[class_idx], 
                nrow=combined_support.shape[1],
                save_path=os.path.join(task_save_dir, f"combined_class{class_idx}_grid.png")
            )
        
        print(f"任务 {task_name} 的图像已保存到 {task_save_dir}")
    
    print("\n所有指定任务的图像已保存")
    
    # 3. 只使用支持集和查询集(不使用LoRA)
    print("\n=== 只使用支持集(不使用LoRA) ===")
    
    no_lora_loader = FSLTaskLoader(
        dataset_path="./task_images",
        n_way=5,
        n_support=5,
        n_query=15,
        include_lora=False  # 不加载LoRA图像
    )
    
    # 获取第一个任务并保存图像
    support_images, support_labels, query_images, query_labels, task_info = no_lora_loader.get_task(0)
    task_name = task_info["task_name"]
    no_lora_dir = os.path.join(save_dir, f"no_lora_{task_name}")
    os.makedirs(no_lora_dir, exist_ok=True)
    
    # 保存第一个类别的支持集和查询集
    save_grid_image(
        support_images[0], 
        nrow=support_images.shape[1],
        save_path=os.path.join(no_lora_dir, "support_class0_grid.png")
    )
    
    save_grid_image(
        query_images[0], 
        nrow=5,
        save_path=os.path.join(no_lora_dir, "query_class0_grid.png")
    )
    
    print(f"不使用LoRA的图像已保存到 {no_lora_dir}")

if __name__ == "__main__":
    simple_example() 