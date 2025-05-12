import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np

class OrderedTaskDataset(Dataset):
    """
    有序任务数据集，用于加载小样本学习任务中的图像
    """
    def __init__(self, dataset_path, task_id=None, n_way=5, n_shot=5, transform=None, image_type="origin"):
        """
        初始化数据集
        
        参数:
        - dataset_path: 数据集根目录路径
        - task_id: 指定任务ID，None表示使用所有任务
        - n_way: 每个任务中的类别数量
        - n_shot: 每个类别的样本数量
        - transform: 图像变换
        - image_type: "origin"或"lora"，指定读取哪类图像
        """
        self.dataset_path = dataset_path
        self.task_id = task_id
        self.n_way = n_way
        self.n_shot = n_shot
        self.transform = transform
        self.image_type = image_type
        
        # 初始化数据结构
        self.tasks = []
        self.task_indices = []
        self.class_indices = {}
        
        # 加载任务和类别信息
        self._load_tasks()
        
        # 定义默认转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
    
    def _load_tasks(self):
        """加载所有任务和类别信息"""
        # 获取任务文件夹列表并按数字排序
        tasks = sorted([d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d))],
                     key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else x)
        
        # 如果指定了任务ID，则只使用该任务
        if self.task_id is not None:
            if isinstance(self.task_id, int):
                tasks = [tasks[self.task_id]] if 0 <= self.task_id < len(tasks) else []
            elif isinstance(self.task_id, str):
                tasks = [self.task_id] if self.task_id in tasks else []
        
        # 处理每个任务
        for task_name in tasks:
            task_path = os.path.join(self.dataset_path, task_name)
            
            # 获取类别文件夹列表
            classes = sorted([d for d in os.listdir(task_path) 
                           if os.path.isdir(os.path.join(task_path, d))])
            
            # 如果类别数量不足，跳过该任务
            if len(classes) < self.n_way:
                print(f"警告: 任务 {task_name} 类别数量不足 ({len(classes)} < {self.n_way})，跳过该任务")
                continue
            
            # 只使用前n_way个类别
            classes = classes[:self.n_way]
            
            task_data = {"name": task_name, "classes": {}}
            task_valid = True
            
            # 处理每个类别
            for class_idx, class_name in enumerate(classes):
                class_path = os.path.join(task_path, class_name)
                
                # 确保有origin/lora子文件夹
                type_path = os.path.join(class_path, self.image_type)
                if not os.path.exists(type_path):
                    print(f"警告: 类别 {class_name} 缺少 {self.image_type} 文件夹，跳过该类别")
                    task_valid = False
                    break
                
                # 获取图像文件列表
                images = sorted([f for f in os.listdir(type_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                
                # 如果图像数量不足，跳过该任务
                if len(images) < self.n_shot:
                    print(f"警告: 类别 {class_name} 图像数量不足 ({len(images)} < {self.n_shot})，跳过该任务")
                    task_valid = False
                    break
                
                # 只使用前n_shot张图像
                images = images[:self.n_shot]
                
                # 保存类别数据
                task_data["classes"][class_name] = {
                    "index": class_idx,
                    "images": [os.path.join(type_path, img) for img in images]
                }
            
            # 如果任务有效，添加到任务列表
            if task_valid:
                self.tasks.append(task_data)
                
                # 更新任务索引
                for class_name, class_data in task_data["classes"].items():
                    for img_idx, img_path in enumerate(class_data["images"]):
                        # 索引格式: (任务索引, 类别名称, 图像索引)
                        self.task_indices.append((len(self.tasks) - 1, class_name, img_idx))
        
        print(f"已加载 {len(self.tasks)} 个有效任务")
    
    def __len__(self):
        """数据集大小 = 任务数 × 类别数 × 每类样本数"""
        return len(self.task_indices)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        # 解析索引
        task_idx, class_name, img_idx = self.task_indices[idx]
        
        # 获取任务和类别数据
        task = self.tasks[task_idx]
        class_data = task["classes"][class_name]
        
        # 获取图像路径和类别索引
        img_path = class_data["images"][img_idx]
        class_idx = class_data["index"]
        
        # 加载和处理图像
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return {
            "image": img,
            "label": class_idx,
            "task_name": task["name"],
            "class_name": class_name,
            "task_idx": task_idx,
            "img_idx": img_idx
        }

class TaskBatchSampler:
    """任务批次采样器，确保每个批次只包含同一个任务的数据"""
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # 按任务分组索引
        self.task_groups = {}
        for i, (task_idx, _, _) in enumerate(dataset.task_indices):
            if task_idx not in self.task_groups:
                self.task_groups[task_idx] = []
            self.task_groups[task_idx].append(i)
    
    def __iter__(self):
        # 任务顺序
        task_indices = sorted(self.task_groups.keys())
        
        # 遍历每个任务
        for task_idx in task_indices:
            indices = self.task_groups[task_idx]
            
            # 分批
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size <= len(indices) or not self.drop_last:
                    batch_indices = indices[i:i + self.batch_size]
                    yield batch_indices
    
    def __len__(self):
        """批次数量"""
        count = 0
        for indices in self.task_groups.values():
            batch_count = (len(indices) + self.batch_size - 1) // self.batch_size
            if self.drop_last and len(indices) % self.batch_size != 0:
                batch_count -= 1
            count += batch_count
        return count

def get_origin_loader(dataset_path, task_id=None, n_way=5, n_shot=5, 
                      transform=None, batch_size=None, num_workers=4):
    """
    创建原始图像数据加载器
    
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
        image_type="support"  # 使用"support"替代"origin"
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

def get_lora_loader(dataset_path, task_id=None, n_way=5, n_shot=5, 
                    transform=None, batch_size=None, num_workers=4):
    """
    创建Lora图像数据加载器
    
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
        image_type="lora"  # 这里是唯一与origin加载器的区别
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

def get_query_loader(dataset_path, task_id=None, n_way=5, n_shot=5, 
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

def load_paired_task_data(dataset_path, task_id=None, n_way=5, n_shot=5, 
                          transform=None, batch_size=None, num_workers=4,
                          image_type_a="support", image_type_b="lora"):
    """
    加载配对的图像数据
    
    参数:
    - dataset_path: 数据集根目录
    - task_id: 指定任务ID, None表示加载所有任务
    - n_way: 类别数量
    - n_shot: 每个类别的样本数量 
    - transform: 图像变换
    - batch_size: 批次大小，默认为 n_way * n_shot
    - num_workers: 数据加载进程数
    - image_type_a: 第一个加载器的图像类型，默认为"support"
    - image_type_b: 第二个加载器的图像类型，默认为"lora"
    
    返回:
    - loader_a: 第一种类型的图像数据加载器
    - loader_b: 第二种类型的图像数据加载器
    """
    # 如果没有指定batch_size，则设为一个任务的全部样本数
    if batch_size is None:
        batch_size = n_way * n_shot
    
    # 创建第一个数据集
    dataset_a = OrderedTaskDataset(
        dataset_path=dataset_path,
        task_id=task_id,
        n_way=n_way,
        n_shot=n_shot,
        transform=transform,
        image_type=image_type_a
    )
    
    # 创建第一个批次采样器
    batch_sampler_a = TaskBatchSampler(
        dataset=dataset_a,
        batch_size=batch_size,
        drop_last=False
    )
    
    # 创建第一个数据加载器
    loader_a = DataLoader(
        dataset=dataset_a,
        batch_sampler=batch_sampler_a,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 创建第二个数据集
    dataset_b = OrderedTaskDataset(
        dataset_path=dataset_path,
        task_id=task_id,
        n_way=n_way,
        n_shot=n_shot,
        transform=transform,
        image_type=image_type_b
    )
    
    # 创建第二个批次采样器
    batch_sampler_b = TaskBatchSampler(
        dataset=dataset_b,
        batch_size=batch_size,
        drop_last=False
    )
    
    # 创建第二个数据加载器
    loader_b = DataLoader(
        dataset=dataset_b,
        batch_sampler=batch_sampler_b,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader_a, loader_b

def get_paired_batches(origin_loader, lora_loader):
    """
    生成配对的origin和lora批次
    
    使用方法:
    for origin_batch, lora_batch in get_paired_batches(origin_loader, lora_loader):
        # 处理配对的批次
    """
    origin_iter = iter(origin_loader)
    lora_iter = iter(lora_loader)
    
    while True:
        try:
            origin_batch = next(origin_iter)
            lora_batch = next(lora_iter)
            yield origin_batch, lora_batch
        except StopIteration:
            break

def format_data_for_fsl(batch_data, n_way, n_shot):
    """
    将批次数据重新格式化为小样本学习所需的格式
    
    输入:
    - batch_data: 批次数据字典
    - n_way: 类别数
    - n_shot: 每类样本数
    
    输出:
    - support_images: 形状为 [n_way, n_shot, channels, height, width]
    - support_labels: 形状为 [n_way, n_shot]
    """
    images = batch_data["image"]
    labels = batch_data["label"]
    
    # 重塑为 [n_way, n_shot, channels, height, width]
    support_images = images.reshape(n_way, n_shot, *images.shape[1:])
    support_labels = labels.reshape(n_way, n_shot)
    
    return support_images, support_labels

def create_dummy_dataset(base_path, n_tasks=3, n_classes=5, n_shots=5):
    """
    创建一个用于测试的虚拟数据集结构
    
    参数:
    - base_path: 数据集根目录路径
    - n_tasks: 任务数量
    - n_classes: 每个任务的类别数量
    - n_shots: 每个类别的样本数量
    """
    os.makedirs(base_path, exist_ok=True)
    
    # 生成一个简单的彩色图像
    def generate_image(color):
        img = Image.new('RGB', (224, 224), color=color)
        return img
    
    # 创建任务
    for task_idx in range(n_tasks):
        task_name = f"task_{task_idx}"
        task_path = os.path.join(base_path, task_name)
        os.makedirs(task_path, exist_ok=True)
        
        # 创建类别
        for class_idx in range(n_classes):
            class_name = f"class_{class_idx}"
            class_path = os.path.join(task_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            # 创建origin和lora文件夹
            for img_type in ["origin", "lora"]:
                type_path = os.path.join(class_path, img_type)
                os.makedirs(type_path, exist_ok=True)
                
                # 生成图像
                for shot_idx in range(n_shots):
                    if img_type == "origin":
                        # 对于origin图像，使用固定颜色
                        color = (50 * class_idx, 30, 30)
                    else:
                        # 对于lora图像，使用不同颜色
                        color = (30, 50 * class_idx, 30)
                    
                    img = generate_image(color)
                    img_name = f"img_{shot_idx}.jpg"
                    img_path = os.path.join(type_path, img_name)
                    img.save(img_path)
    
    print(f"已创建虚拟数据集于 {base_path}，包含 {n_tasks} 个任务")

def load_complete_task_data(dataset_path, task_id=None, n_way=5, n_support=5, n_query=15,
                           transform=None, batch_size_support=None, batch_size_query=None, num_workers=4,
                           use_lora_as_origin=False):
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
    - use_lora_as_origin: 是否使用lora图像替代缺失的support图像
    
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
    if use_lora_as_origin:
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