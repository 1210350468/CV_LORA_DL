import torch
from torch.utils.data import DataLoader
import os

from .OrderedTaskLoader import (
    OrderedTaskDataset,
    TaskBatchSampler,
    get_origin_loader,
    get_lora_loader,
    get_query_loader,
    load_paired_task_data,
    load_complete_task_data,
    format_data_for_fsl
)

class FSLBatchIterator:
    """自动格式化的小样本学习批次迭代器"""
    def __init__(self, dataloader, n_way, n_shot):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.n_way = n_way
        self.n_shot = n_shot
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
            # 自动格式化
            images, labels = format_data_for_fsl(batch, self.n_way, self.n_shot)
            # 添加任务信息
            task_info = {
                "task_name": batch["task_name"][0],
                "original_batch": batch
            }
            return images, labels, task_info
        except StopIteration:
            raise StopIteration
            
class FSLDataLoader:
    """
    自动格式化的小样本学习数据加载器
    
    自动处理OrderedTaskLoader中的数据格式化问题，
    允许直接获取格式化后的数据[n_way, n_shot, C, H, W]。
    """
    def __init__(self, 
                 dataset_path, 
                 n_way=5, 
                 n_shot=5, 
                 image_type="support", 
                 transform=None, 
                 batch_size=None, 
                 num_workers=4):
        """
        初始化FSL数据加载器
        
        参数:
        - dataset_path: 数据集根目录
        - n_way: 类别数量
        - n_shot: 每类样本数
        - image_type: 图像类型，可选"support"、"lora"或"query"
        - transform: 图像变换
        - batch_size: 批次大小，默认为n_way*n_shot
        - num_workers: 数据加载线程数
        """
        self.dataset_path = dataset_path
        self.n_way = n_way
        self.n_shot = n_shot
        self.image_type = image_type
        self.transform = transform
        self.batch_size = batch_size or (n_way * n_shot)
        self.num_workers = num_workers
        
        # 创建基础数据加载器
        if image_type == "support":
            self.base_loader = get_origin_loader(
                dataset_path=dataset_path,
                n_way=n_way,
                n_shot=n_shot,
                transform=transform,
                batch_size=self.batch_size,
                num_workers=num_workers
            )
        elif image_type == "lora":
            self.base_loader = get_lora_loader(
                dataset_path=dataset_path,
                n_way=n_way,
                n_shot=n_shot,
                transform=transform,
                batch_size=self.batch_size,
                num_workers=num_workers
            )
        elif image_type == "query":
            self.base_loader = get_query_loader(
                dataset_path=dataset_path,
                n_way=n_way,
                n_shot=n_shot,
                transform=transform,
                batch_size=self.batch_size,
                num_workers=num_workers
            )
        else:
            raise ValueError(f"不支持的图像类型: {image_type}，请使用'support'、'lora'或'query'")
    
    def __iter__(self):
        """返回自动格式化的迭代器"""
        return FSLBatchIterator(self.base_loader, self.n_way, self.n_shot)
    
    def __len__(self):
        """加载器中的批次数量"""
        return len(self.base_loader)
    
    def raw_loader(self):
        """获取原始的DataLoader"""
        return self.base_loader

class FSLTaskLoader:
    """
    小样本学习任务加载器
    
    同时加载支持集和查询集，自动格式化数据
    """
    def __init__(self, 
                 dataset_path, 
                 n_way=5, 
                 n_support=5, 
                 n_query=15, 
                 transform=None, 
                 num_workers=4,
                 use_lora_as_support=False,
                 include_lora=True):
        """
        初始化FSL任务加载器
        
        参数:
        - dataset_path: 数据集根目录
        - n_way: 类别数量
        - n_support: 每类支持集样本数
        - n_query: 每类查询集样本数
        - transform: 图像变换
        - num_workers: 数据加载线程数
        - use_lora_as_support: 是否使用lora替代support
        - include_lora: 是否包含lora图像
        """
        self.dataset_path = dataset_path
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.transform = transform
        self.num_workers = num_workers
        self.use_lora_as_support = use_lora_as_support
        self.include_lora = include_lora
        
        # 加载数据
        if include_lora:
            self.support_loader, self.lora_loader, self.query_loader = load_complete_task_data(
                dataset_path=dataset_path,
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                transform=transform,
                num_workers=num_workers,
                use_lora_as_origin=use_lora_as_support
            )
        else:
            # 仅加载支持集和查询集
            self.support_loader = get_origin_loader(
                dataset_path=dataset_path,
                n_way=n_way,
                n_shot=n_support,
                transform=transform,
                num_workers=num_workers
            )
            self.lora_loader = None
            self.query_loader = get_query_loader(
                dataset_path=dataset_path,
                n_way=n_way,
                n_shot=n_query,
                transform=transform,
                num_workers=num_workers
            )
        
        # 创建迭代器
        self.support_iter = FSLBatchIterator(self.support_loader, n_way, n_support)
        if include_lora:
            self.lora_iter = FSLBatchIterator(self.lora_loader, n_way, n_support)
        else:
            self.lora_iter = None
        self.query_iter = FSLBatchIterator(self.query_loader, n_way, n_query)
    
    def __iter__(self):
        """
        返回任务迭代器
        
        每次迭代返回一个任务的(支持集, 查询集)或(支持集, LoRA集, 查询集)
        """
        self.support_iter = FSLBatchIterator(self.support_loader, self.n_way, self.n_support)
        if self.include_lora:
            self.lora_iter = FSLBatchIterator(self.lora_loader, self.n_way, self.n_support)
        self.query_iter = FSLBatchIterator(self.query_loader, self.n_way, self.n_query)
        
        return self
    
    def __next__(self):
        try:
            support_images, support_labels, support_info = next(self.support_iter)
            query_images, query_labels, query_info = next(self.query_iter)
            
            task_info = {
                "task_name": support_info["task_name"],
                "support_info": support_info,
                "query_info": query_info
            }
            
            if self.include_lora:
                lora_images, lora_labels, lora_info = next(self.lora_iter)
                task_info["lora_info"] = lora_info
                return support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info
            else:
                return support_images, support_labels, query_images, query_labels, task_info
        except StopIteration:
            raise StopIteration
    
    def __len__(self):
        """加载器中的任务数量"""
        return min(len(self.support_loader), len(self.query_loader))
    
    def get_task(self, task_idx):
        """
        获取指定索引的任务
        
        参数:
        - task_idx: 任务索引
        
        返回:
        任务数据，格式同__next__方法
        """
        if task_idx >= len(self):
            raise IndexError(f"任务索引{task_idx}超出范围(0-{len(self)-1})")
        
        # 重置迭代器
        support_iter = iter(self.support_loader)
        query_iter = iter(self.query_loader)
        if self.include_lora:
            lora_iter = iter(self.lora_loader)
        
        # 前进到指定索引
        for _ in range(task_idx):
            next(support_iter)
            next(query_iter)
            if self.include_lora:
                next(lora_iter)
        
        # 获取数据
        support_batch = next(support_iter)
        query_batch = next(query_iter)
        
        # 格式化数据
        support_images, support_labels = format_data_for_fsl(support_batch, self.n_way, self.n_support)
        query_images, query_labels = format_data_for_fsl(query_batch, self.n_way, self.n_query)
        
        task_info = {
            "task_name": support_batch["task_name"][0],
            "support_batch": support_batch,
            "query_batch": query_batch
        }
        
        if self.include_lora:
            lora_batch = next(lora_iter)
            lora_images, lora_labels = format_data_for_fsl(lora_batch, self.n_way, self.n_support)
            task_info["lora_batch"] = lora_batch
            return support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info
        else:
            return support_images, support_labels, query_images, query_labels, task_info

def get_combined_support(support_images, lora_images=None):
    """
    合并支持集和LoRA图像
    
    参数:
    - support_images: 支持集图像 [n_way, n_shot, C, H, W]
    - lora_images: LoRA图像 [n_way, n_shot, C, H, W]，如果为None则返回support_images
    
    返回:
    - combined: 合并后的图像 [n_way, n_shot*2, C, H, W]
    """
    if lora_images is None:
        return support_images
    
    return torch.cat([support_images, lora_images], dim=1) 