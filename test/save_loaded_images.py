import os
import torch
import sys
import numpy as np
from torchvision import utils
from torchvision.transforms import functional as TF
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.OrderedTaskLoader import (
    load_complete_task_data,
    format_data_for_fsl
)

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
    img = TF.to_pil_image(img)
    # 保存
    img.save(save_path)

def save_grid_image(tensors, nrow, save_path):
    """将多个tensor创建为网格图像并保存"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建网格
    grid = utils.make_grid(tensors, nrow=nrow, padding=2, normalize=True)
    # 转换为PIL图像并保存
    img = TF.to_pil_image(grid)
    img.save(save_path)

def save_loaded_images():
    """加载并保存图像示例"""
    # 设置图像保存目录
    save_dir = "./saved_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置数据路径
    dataset_path = "./task_images"
    if not os.path.exists(dataset_path):
        print(f"错误: 目录 {dataset_path} 不存在")
        return
    
    # 参数设置
    n_way = 5  # 类别数量
    n_support = 5  # 每个类别的支持集样本数
    n_query = 15  # 每个类别的查询集样本数
    
    print(f"正在加载数据...")
    # 加载数据
    try:
        support_loader, lora_loader, query_loader = load_complete_task_data(
            dataset_path=dataset_path,
            n_way=n_way,
            n_support=n_support,
            n_query=n_query
        )
        
        print(f"数据加载器已创建:")
        print(f"- support_loader 批次数={len(support_loader)}")
        print(f"- lora_loader 批次数={len(lora_loader)}")
        print(f"- query_loader 批次数={len(query_loader)}")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"开始保存图像...")
    
    # 迭代器
    support_iter = iter(support_loader)
    lora_iter = iter(lora_loader)
    query_iter = iter(query_loader)
    
    # 处理前5个任务
    for task_idx in range(min(5, len(support_loader), len(query_loader))):
        try:
            # 获取批次数据
            support_batch = next(support_iter)
            lora_batch = next(lora_iter)
            query_batch = next(query_iter)
            
            task_name = support_batch["task_name"][0]
            print(f"处理任务 {task_idx+1}: {task_name}")
            
            # 创建当前任务的保存目录
            task_save_dir = os.path.join(save_dir, f"task_{task_idx}_{task_name}")
            os.makedirs(task_save_dir, exist_ok=True)
            
            # 格式化数据
            support_images, support_labels = format_data_for_fsl(support_batch, n_way, n_support)
            lora_images, lora_labels = format_data_for_fsl(lora_batch, n_way, n_support)
            query_images, query_labels = format_data_for_fsl(query_batch, n_way, n_query)
            
            # 1. 保存每个类别的第一张图片样本
            for class_idx in range(n_way):
                # 支持集
                save_tensor_as_image(
                    support_images[class_idx, 0], 
                    os.path.join(task_save_dir, f"support_class{class_idx}_sample0.png")
                )
                
                # LoRA集
                save_tensor_as_image(
                    lora_images[class_idx, 0], 
                    os.path.join(task_save_dir, f"lora_class{class_idx}_sample0.png")
                )
                
                # 查询集
                save_tensor_as_image(
                    query_images[class_idx, 0], 
                    os.path.join(task_save_dir, f"query_class{class_idx}_sample0.png")
                )
            
            # 2. 保存每种类型的网格图像
            # 支持集网格（所有类别）
            for class_idx in range(n_way):
                save_grid_image(
                    support_images[class_idx], 
                    nrow=n_support,
                    save_path=os.path.join(task_save_dir, f"support_class{class_idx}_grid.png")
                )
                
                save_grid_image(
                    lora_images[class_idx], 
                    nrow=n_support,
                    save_path=os.path.join(task_save_dir, f"lora_class{class_idx}_grid.png")
                )
                
                save_grid_image(
                    query_images[class_idx], 
                    nrow=5,  # 只显示前5张查询图像
                    save_path=os.path.join(task_save_dir, f"query_class{class_idx}_grid.png")
                )
            
            # 3. 保存支持集和LoRA集的合并
            combined_images = torch.cat([support_images, lora_images], dim=1)
            for class_idx in range(n_way):
                save_grid_image(
                    combined_images[class_idx], 
                    nrow=n_support*2,
                    save_path=os.path.join(task_save_dir, f"combined_class{class_idx}_grid.png")
                )
                
            print(f"已保存任务 {task_name} 的图像")
            
        except StopIteration:
            print("所有任务数据已处理完毕")
            break
        except Exception as e:
            print(f"处理任务时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"所有图像已保存到目录: {save_dir}")

if __name__ == "__main__":
    save_loaded_images() 