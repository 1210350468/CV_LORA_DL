import os
import torch
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.OrderedTaskLoader import (
    OrderedTaskDataset, 
    TaskBatchSampler,
    load_paired_task_data,
    get_paired_batches, 
    format_data_for_fsl
)
from torch.utils.data import DataLoader

def visualize_batch(images, labels, title=None, nrow=5):
    """可视化一个批次的图像"""
    # 如果是张量，转换为numpy数组
    if torch.is_tensor(images):
        if images.dim() == 4:  # [batch, channels, height, width]
            grid = utils.make_grid(images, nrow=nrow)
            npimg = grid.numpy().transpose((1, 2, 0))
        elif images.dim() == 5:  # [n_way, n_shot, channels, height, width]
            # 重塑为 [n_way*n_shot, channels, height, width]
            flat_images = images.reshape(-1, *images.shape[2:])
            grid = utils.make_grid(flat_images, nrow=images.shape[1])
            npimg = grid.numpy().transpose((1, 2, 0))
        plt.imshow(npimg)
    else:
        plt.imshow(images)
    
    # 设置标题
    if title:
        plt.title(title)
    else:
        if torch.is_tensor(labels):
            if labels.dim() == 1:  # [batch]
                plt.title(f"Labels: {labels.tolist()}")
            elif labels.dim() == 2:  # [n_way, n_shot]
                plt.title(f"Labels shape: {labels.shape}")
    
    plt.axis('off')
    plt.show()

def get_query_loader(dataset_path, task_id=None, n_way=5, n_shot=15, 
                    transform=None, batch_size=None, num_workers=4):
    """
    创建查询集数据加载器
    
    参数:
    - dataset_path: 数据集根目录路径
    - task_id: 指定任务ID
    - n_way: 每个任务中的类别数量
    - n_shot: 每个类别的样本数量
    - transform: 图像变换
    - batch_size: 批次大小，默认为 n_way * n_shot
    - num_workers: 数据加载进程数
    """
    # 如果没有指定batch_size，则设为一个任务的全部样本数
    if batch_size is None:
        batch_size = n_way * n_shot
    
    # 创建数据集
    dataset = OrderedTaskDataset(
        dataset_path=dataset_path,
        task_id=task_id,
        n_way=n_way,
        n_shot=n_shot,
        transform=transform,
        image_type="query"  # 使用"query"类型
    )
    
    # 创建批次采样器
    batch_sampler = TaskBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def load_complete_task_data(dataset_path, task_id=None, n_way=5, n_support=5, n_query=15,
                          transform=None, batch_size_support=None, batch_size_query=None, num_workers=4,
                          use_lora_as_support=False):
    """
    加载完整任务数据，包括支持集和查询集
    
    参数:
    - dataset_path: 数据集根目录
    - task_id: 指定任务ID，None表示加载所有任务
    - n_way: 类别数
    - n_support: 每个类别的支持集样本数
    - n_query: 每个类别的查询集样本数
    - transform: 图像变换
    - batch_size_support: 支持集批次大小
    - batch_size_query: 查询集批次大小
    - num_workers: 数据加载进程数
    - use_lora_as_support: 是否使用lora图像替代缺失的support图像
    
    返回:
    - support_loader: 原始支持集加载器
    - lora_loader: Lora支持集加载器
    - query_loader: 查询集加载器
    """
    # 设置默认批次大小
    if batch_size_support is None:
        batch_size_support = n_way * n_support
    if batch_size_query is None:
        batch_size_query = n_way * n_query
    
    # 加载支持集
    if use_lora_as_support:
        # 使用lora图像替代support图像
        _, lora_loader = load_paired_task_data(
            dataset_path=dataset_path,
            task_id=task_id,
            n_way=n_way,
            n_shot=n_support,
            transform=transform,
            batch_size=batch_size_support,
            num_workers=num_workers,
            image_type_a="lora",
            image_type_b="lora"
        )
        # 把lora_loader用作support_loader
        support_loader = lora_loader
    else:
        # 正常加载support和lora
        support_loader, lora_loader = load_paired_task_data(
            dataset_path=dataset_path,
            task_id=task_id,
            n_way=n_way,
            n_shot=n_support,
            transform=transform,
            batch_size=batch_size_support,
            num_workers=num_workers
        )
    
    # 加载查询集
    query_loader = get_query_loader(
        dataset_path=dataset_path,
        task_id=task_id,
        n_way=n_way,
        n_shot=n_query,
        transform=transform,
        batch_size=batch_size_query,
        num_workers=num_workers
    )
    
    return support_loader, lora_loader, query_loader

def test_complete_dataset():
    """测试加载完整的真实数据集，包括支持集和查询集"""
    # 使用实际的数据路径
    dataset_path = "./task_images"
    
    # 检查目录是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 目录 {dataset_path} 不存在")
        return
    
    # 显示目录结构
    print(f"数据目录结构:")
    task_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"发现 {len(task_dirs)} 个任务: {', '.join(task_dirs[:5])}{'...' if len(task_dirs) > 5 else ''}")
    
    if not task_dirs:
        print("未找到任务目录")
        return
    
    # 检查第一个任务的结构
    sample_task = task_dirs[0]
    print(f"\n任务 {sample_task} 中的类别:")
    class_dirs = [d for d in os.listdir(os.path.join(dataset_path, sample_task)) 
                 if os.path.isdir(os.path.join(dataset_path, sample_task, d))]
    print(f"发现 {len(class_dirs)} 个类别: {', '.join(class_dirs)}")
    
    if not class_dirs:
        print("未找到类别目录")
        return
    
    # 检查图像类型
    sample_class = class_dirs[0]
    print(f"\n类别 {sample_class} 中的图像类型:")
    type_dirs = [d for d in os.listdir(os.path.join(dataset_path, sample_task, sample_class)) 
                if os.path.isdir(os.path.join(dataset_path, sample_task, sample_class, d))]
    print(f"发现类型: {', '.join(type_dirs)}")
    
    # 检查是否有support文件夹
    has_support = False
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, sample_task, class_dir)
        if os.path.isdir(os.path.join(class_path, "support")):
            has_support = True
            break
    
    if not has_support:
        print("\n注意: 未找到support文件夹，将使用lora图像替代support图像")
    
    # 设置参数
    n_way = 5  # 类别数量 
    n_support = 5  # 每个类别的支持集样本数
    n_query = 15  # 每个类别的查询集样本数
    
    # 加载数据
    print("\n正在加载数据...")
    try:
        support_loader, lora_loader, query_loader = load_complete_task_data(
            dataset_path=dataset_path,
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            use_lora_as_support=not has_support  # 如果没有support文件夹，使用lora图像替代
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
    
    # 测试数据加载和对齐
    print("\n开始测试数据加载和对齐...")
    
    # 转换为迭代器
    support_iter = iter(support_loader)
    lora_iter = iter(lora_loader)
    query_iter = iter(query_loader)
    
    # 逐任务处理
    for i in range(min(len(support_loader), len(query_loader))):
        try:
            # 获取批次数据
            support_batch = next(support_iter)
            lora_batch = next(lora_iter)
            query_batch = next(query_iter)
            
            print(f"\n处理任务 {i+1}/{min(len(support_loader), len(query_loader))}")
            
            # 检查任务名称
            support_task = support_batch["task_name"][0]
            lora_task = lora_batch["task_name"][0]
            query_task = query_batch["task_name"][0]
            
            print(f"任务名称: support={support_task}, lora={lora_task}, query={query_task}")
            
            # 检查标签
            support_labels = support_batch["label"]
            lora_labels = lora_batch["label"]
            query_labels = query_batch["label"]
            
            print(f"Support标签形状: {support_labels.shape}")
            print(f"Lora标签形状: {lora_labels.shape}")
            print(f"Query标签形状: {query_labels.shape}")
            
            # 将数据转换为小样本学习格式
            support_images, support_labels = format_data_for_fsl(support_batch, n_way, n_support)
            lora_support, lora_labels = format_data_for_fsl(lora_batch, n_way, n_support)
            query_images, query_labels = format_data_for_fsl(query_batch, n_way, n_query)
            
            print(f"\n格式化后的数据形状:")
            print(f"- Support支持集: {support_images.shape}")
            print(f"- Lora支持集: {lora_support.shape}")
            print(f"- 查询集: {query_images.shape}")
            
            # 可视化数据样本
            print("\n正在可视化数据样本...")
            
            # 显示每种类型的第一行图像
            print("Support支持集:")
            visualize_batch(support_images[0], support_labels[0], 
                          title=f"任务 {i+1} - Support支持集 (第一个类别)")
            
            print("Lora支持集:")
            visualize_batch(lora_support[0], lora_labels[0], 
                          title=f"任务 {i+1} - Lora支持集 (第一个类别)")
            
            print("查询集:")
            visualize_batch(query_images[0], query_labels[0], 
                          title=f"任务 {i+1} - 查询集 (第一个类别)")
            
            # 将支持集堆叠在一起
            combined_support = torch.cat([support_images, lora_support], dim=1)
            combined_labels = torch.cat([support_labels, lora_labels], dim=1)
            
            print(f"\n堆叠后的支持集形状: {combined_support.shape}")
            
            # 展示组合的支持集
            print("组合的支持集 (Support + Lora):")
            visualize_batch(combined_support[0], combined_labels[0], 
                          title=f"任务 {i+1} - 组合支持集 (第一个类别)")
            
            # 只测试前两个任务
            if i >= 1:
                break
                
        except StopIteration:
            print("所有任务数据已处理完毕")
            break
        except Exception as e:
            print(f"处理任务时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n测试完成!")

def model_training_example():
    """展示如何在训练中使用数据加载器"""
    print("模型训练示例代码:")
    print("""
    # 1. 加载数据
    support_loader, lora_loader, query_loader = load_complete_task_data(
        dataset_path="./task_images",
        n_way=5,
        n_support=5,
        n_query=15
    )
    
    # 2. 准备模型
    model = ... # 创建或加载模型
    
    # 3. 对每个任务进行微调
    for task_idx in range(min(len(support_loader), len(query_loader))):
        # 获取任务数据
        support_batch = next(iter(support_loader))
        lora_batch = next(iter(lora_loader))
        query_batch = next(iter(query_loader))
        
        # 格式化数据
        support_images, _ = format_data_for_fsl(support_batch, n_way=5, n_shot=5)
        lora_support, _ = format_data_for_fsl(lora_batch, n_way=5, n_shot=5)
        query_images, query_labels = format_data_for_fsl(query_batch, n_way=5, n_shot=15)
        
        # 合并支持集
        combined_support = torch.cat([support_images, lora_support], dim=1)
        
        # 微调模型
        model.fine_tune(
            combined_support,  # [5, 10, 3, 224, 224]
            query_images,      # [5, 15, 3, 224, 224]
            query_labels       # [5, 15]
        )
    """)

if __name__ == "__main__":
    test_complete_dataset()
    model_training_example() 