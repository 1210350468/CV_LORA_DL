import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# 假设 CustomResNet 和元学习模型定义在另一个文件中
from models.Pretrain_ResNet import CustomResNet, get_resnet_model
from models.ProtoNet import MetaLearningModel, get_meta_model
from dataloader.Metastage_Dataloader import get_meta_dataloader  # 导入数据加载器

def train_meta_learning(model, meta_loader, optimizer, num_epochs=20, save_path='./output/meta_learning/'):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = model.cuda()  # 确保模型在 GPU 上
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # 遍历每一个元任务
        for batch_idx, (images, labels) in enumerate(tqdm(meta_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            n_way, total_samples, channels, size, _ = images.size()
            support_shot = total_samples // 4  # 每个任务的支持集样本数
            
            # 分离 support 和 query 数据集
            support_images = images[:, :support_shot, :, :, :].reshape(n_way * support_shot, channels, size, size).cuda()
            query_images = images[:, support_shot:, :, :, :].reshape(n_way * support_shot, channels, size, size).cuda()
            support_labels = labels[:, :support_shot].reshape(-1).cuda()  # 支持集的标签
            query_labels = labels[:, support_shot:].reshape(-1).cuda()    # 查询集的标签

            # 前向传播
            dists = model(support_images, query_images, support_labels, n_way)

            # 计算损失
            loss = F.cross_entropy(dists, query_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(meta_loader)}")
        
        # 每 10 个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{save_path}meta_model_epoch_{epoch+1}.pth')
            print(f"Model saved at {save_path}meta_model_epoch_{epoch+1}.pth")
        
        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in meta_loader:
                n_way, total_samples, channels, size, _ = images.size()
                support_shot = total_samples // 4
                
                # 分离 support 和 query 数据集
                support_images = images[:, :support_shot, :, :, :].reshape(n_way * support_shot, channels, size, size).cuda()
                query_images = images[:, support_shot:, :, :, :].reshape(n_way * support_shot, channels, size, size).cuda()
                query_labels = labels[:, support_shot:].reshape(-1).cuda()  # 查询集的标签

                # 前向传播
                dists = model(support_images, query_images, support_labels=None, num_classes=n_way)
                _, predicted = torch.min(dists, dim=1)  # 预测类别
                correct += (predicted == query_labels).sum().item()
                total += query_labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # 假设你已经通过 get_resnet_model() 得到预训练的 ResNet 模型
    backbone = get_resnet_model(version='resnet18', num_classes=1000).resnet
    meta_model = get_meta_model(backbone=backbone, num_classes=5)

    # 优化器
    optimizer = optim.Adam(meta_model.parameters(), lr=3e-4)

    # 假设你已经定义了一个 meta_loader，它的输出是 [n_way, support + query, channels, size, size]
    meta_loader = get_meta_dataloader()  # 替换为你实际的数据加载器

    # 启动元学习训练
    train_meta_learning(meta_model, meta_loader, optimizer, num_epochs=20)
