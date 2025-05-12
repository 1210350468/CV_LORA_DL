import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import random
import json
import numpy as np
import os
from torchvision import transforms
import torchvision.transforms.functional as TF

# 这个类对应标准加载器中的TransformLoader，用于解析并组合各种转换
class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type == 'RandomResizedCrop':
            return transforms.RandomResizedCrop(self.image_size)
        elif transform_type == 'CenterCrop':
            return transforms.CenterCrop(self.image_size)
        elif transform_type == 'Resize':
            return transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type == 'ImageJitter':
            return ImageJitter(self.jitter_param)
        elif transform_type == 'RandomHorizontalFlip':
            return transforms.RandomHorizontalFlip()
        elif transform_type == 'ToTensor':
            return transforms.ToTensor()
        elif transform_type == 'Normalize':
            return transforms.Normalize(**self.normalize_param)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
    
    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
        
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

# 这个类实现了数据增强中的抖动效果
class ImageJitter(object):
    def __init__(self, jitter_param):
        self.jitter_param = jitter_param
    
    def __call__(self, img):
        brightness = random.uniform(max(0, 1-self.jitter_param['Brightness']), 1+self.jitter_param['Brightness'])
        contrast = random.uniform(max(0, 1-self.jitter_param['Contrast']), 1+self.jitter_param['Contrast'])
        color = random.uniform(max(0, 1-self.jitter_param['Color']), 1+self.jitter_param['Color'])
        
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, color)
        
        return img

# 对应标准加载器中的SubDataset，用于处理单个类别的图像
class SubDataset(Dataset):
    def __init__(self, images_paths, label, transform=None):
        self.images_paths = images_paths
        self.label = label  # 这个是类别索引号
        self.transform = transform
    
    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.label
    
    def __len__(self):
        return len(self.images_paths)

# 对应标准加载器中的SetDataset，用于管理所有类别的数据
class SetDataset(Dataset):
    def __init__(self, json_file, batch_size, transform=None):
        self.transform = transform
        self.batch_size = batch_size
        
        # 从JSON读取数据
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 整理数据到类别字典中
        self.sub_meta = {}
        self.cl_list = []
        
        # 为每个类别创建一个列表
        for item in data:
            label = item["label"]
            if label not in self.sub_meta:
                self.sub_meta[label] = []
                self.cl_list.append(label)
        
        # 添加图片路径到对应类别列表
        for item in data:
            self.sub_meta[item["label"]].append(item["image_path"])
        
        # 打印每个类别的样本数量
        for key in self.cl_list:
            print(f"类别 {key}: {len(self.sub_meta[key])} 个样本")
        
        # 为每个类别创建子数据加载器
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 使用主线程
            pin_memory=False
        )
        
        # 为每个类别创建一个子数据集和数据加载器
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))
    
    def __getitem__(self, i):
        # 返回指定类别的一个batch数据
        return next(iter(self.sub_dataloader[i]))
    
    def __len__(self):
        # 返回类别数量
        return len(self.sub_dataloader)

# 这个类与标准加载器中的EpisodicBatchSampler完全一致
class EpisodicBatchSampler(Sampler):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes  # 总类别数
        self.n_way = n_way          # 每个任务中的类别数
        self.n_episodes = n_episodes  # 总任务数
    
    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i in range(self.n_episodes):
            # 随机选择n_way个类别
            yield torch.randperm(self.n_classes)[:self.n_way]

# 对应标准加载器中的SetDataManager
class MetaDataManager:
    def __init__(self, json_file, image_size, n_way=5, n_support=5, n_query=16, n_episodes=100):
        self.json_file = json_file
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episodes = n_episodes
        
        # 创建转换加载器
        self.trans_loader = TransformLoader(image_size)
    
    def get_data_loader(self, aug=False):
        # 获取转换
        transform = self.trans_loader.get_composed_transform(aug)
        
        # 创建数据集
        dataset = SetDataset(self.json_file, self.batch_size, transform)
        
        # 创建采样器
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episodes)
        
        # 创建数据加载器
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=12,
            pin_memory=True
        )
        
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

# 简化的接口函数，类似于标准示例中的接口
def get_meta_dataloader(json_file, n_way=5, n_support=5, n_query=16, n_episodes=100, image_size=224, aug=False):
    data_manager = MetaDataManager(
        json_file=json_file,
        image_size=image_size,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_episodes
    )
    return data_manager.get_data_loader(aug=aug)

# Test code
if __name__ == "__main__":
    np.random.seed(10)
    json_file = '../utils/EuroSAT.json'
    n_way = 5      # 5-way
    n_support = 5  # 5-shot
    n_query = 15   # 15 query
    n_episodes = 200  # 每个epoch的任务数量
    image_size = 224  # 图像大小

    # 获取数据加载器
    meta_loader = get_meta_dataloader(
        json_file=json_file,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_episodes,
        image_size=image_size
    )
    
    # 测试数据加载器
    for i, (x, y) in enumerate(meta_loader):
        print(f"Task {i+1}:")
        print(f"x shape: {x.shape}")  # 应该是 [n_way, n_support+n_query, 3, image_size, image_size]
        print(f"y shape: {y.shape}")  # 应该是 [n_way, n_support+n_query]
        # 仅测试一个任务
        print("y:",y)
        break
