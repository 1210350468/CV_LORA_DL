import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from torchvision import transforms
import numpy as np
class PretrainDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_info = self.data[idx]
        img_path = img_info['image_path']
        label = img_info['label']
        
        # 加载图像并转换为 RGB 格式
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_pretrain_dataloader(batch_size, json_file, shuffle=True, num_workers=16, aug=False):
    if aug:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 先resize到稍大尺寸
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 更大的裁剪范围
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 增大旋转角度
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 添加颜色抖动
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加平移
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5),  # 添加随机擦除
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = PretrainDataset(json_file=json_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True)
    
    return dataloader
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)