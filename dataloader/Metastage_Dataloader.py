import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import random
import json
from torchvision import transforms

class MetaDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.transform = transform
        self.class_to_images = self._load_class_images()
    
    def _load_class_images(self):
        """将每个类别对应的图片路径加载到字典中"""
        class_to_images = {}
        for item in self.data:
            label = item["label"]
            image_path = item["image_path"]
            if label not in class_to_images:
                class_to_images[label] = []
            class_to_images[label].append(image_path)
        return class_to_images
    
    def __len__(self):
        return sum(len(imgs) for imgs in self.class_to_images.values())  # 返回图片总数
    
    def __getitem__(self, index):
        image_path, label = index
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TaskSampler(Sampler):
    def __init__(self, class_to_images, num_classes, num_support, num_query, task_num):
        self.class_to_images = class_to_images
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.task_num = task_num
        self.classes = list(class_to_images.keys())  # 类别列表
    
    def __iter__(self):
        for _ in range(self.task_num):
            sampled_classes = random.sample(self.classes, self.num_classes)
            task = []
            for cls in sampled_classes:
                images = self.class_to_images[cls]
                selected_images = random.sample(images, self.num_support + self.num_query)
                task.extend([(img, cls) for img in selected_images])
            yield task  # 返回一个任务的采样结果
    
    def __len__(self):
        return self.task_num


# Define a custom collate function to handle the reshaping inside DataLoader
def meta_collate_fn(batch):
    """batch 是一个任务的所有 (image, label) 样本, 在这里处理 reshaping."""
    images, labels = zip(*batch)
    
    images = torch.stack(images)  # 将所有 image 转换为 Tensor
    labels = torch.tensor(labels)  # 将标签转换为 Tensor
    
    # 假设 batch 中每个任务都由 n_way * (n_support + n_query) 张图片组成
    num_classes = len(set(labels.tolist()))  # 根据标签的唯一数量判断 n_way
    num_samples_per_class = len(labels) // num_classes

    # Reshape images to [n_way, n_support + n_query, 3, size, size]
    images = images.view(num_classes, num_samples_per_class, *images.shape[1:])
    labels = labels.view(num_classes, num_samples_per_class)

    return images, labels


def get_meta_dataloader(json_file, num_classes, num_support, num_query, task_num=100, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MetaDataset(json_file=json_file, transform=transform)
    
    # 使用自定义采样器来生成任务
    task_sampler = TaskSampler(dataset.class_to_images, num_classes, num_support, num_query, task_num)
    
    # 传入自定义的 collate_fn
    dataloader = DataLoader(dataset, batch_sampler=task_sampler, num_workers=num_workers, collate_fn=meta_collate_fn,pin_memory=False)
    
    return dataloader


# Test code
if __name__ == "__main__":
    json_file = '../utils/EuroSAT.json'
    num_classes = 5  # 5-way
    num_support = 5  # 5-shot
    num_query = 15   # 15 query
    task_num = 200   # 每个 epoch 的任务数量

    meta_loader = get_meta_dataloader(json_file=json_file, num_classes=num_classes, num_support=num_support, num_query=num_query, task_num=task_num)

    for i, (task_images, task_labels) in enumerate(meta_loader):
        # task_images 的形状会是 [num_classes, num_support + num_query, 3, 84, 84]
        print(f"Task {i + 1}:")
        print(f"Images shape: {task_images.shape}")  # 这应该输出 [n_way, n_support + n_query, 3, size, size]
        print(f"Labels shape: {task_labels.shape}")  # 这将输出 [n_way, n_support + n_query] 或 [n_way, total_samples_per_class]
        break  # 只测试一个 batch
