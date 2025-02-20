import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from torch.cuda.amp import autocast, GradScaler  # 混合精度
import torchvision.models as models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 导入模型和数据加载器
from dataloader.Pretrain_Dataloader import get_pretrain_dataloader
from models.Pretrain_VIT import get_vit_model
from torch.cuda.amp import GradScaler, autocast
# 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, save_path='./output/pretrain/VIT', accumulation_steps=4,scheduler=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # model = model.cuda()  # 确保模型在 GPU 上
    scaler = GradScaler()  # 初始化 GradScaler

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # 使用 tqdm 包装 data_loader 以显示进度条
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # 前向传播和反向传播，使用混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        # 每个 epoch 调整学习率
        # scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # 每 50 个 epoch 保存一次模型
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'{save_path}modelVIT_{epoch+1}.pth')
            print(f"Model saved at {save_path}model_{epoch+1}.pth")

        # 每10轮进行验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            print("Testing...")
            with torch.no_grad():
                acc = []
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    with autocast():
                        outputs = model(inputs)
                    acc.append((outputs.argmax(dim=1) == labels).float().mean().item())

                print(f"Epoch [{epoch+1}/{num_epochs}], Acc: {sum(acc) / len(acc) * 100} %")
                with open(f'{save_path}log_VIT.txt', 'a') as f:
                    f.write(f"Epoch [{epoch+1}/{num_epochs}], Acc: {sum(acc) / len(acc) * 100} % \n")

if __name__ == "__main__":
    # 选择ViT模型并加载预训练权重
    # model = get_vit_model(num_classes=100, pretrained=True)  # 使用预训练权重
    model = models.vit_b_16(pretrained=True)
    num_classes = 100  # 假设有 100 个类别
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.cuda()

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.05)  # 使用AdamW和较高的Weight Decay
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)  # Cosine Annealing学习率调度
    criterion = nn.CrossEntropyLoss()

    # 准备数据
    train_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImage_train.json')
    val_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImage_val.json')

    # 启动训练
    # train_model(model, train_loader,val_loader,optimizer, scheduler, criterion, num_epochs=100)
    train_model(model, train_loader,val_loader,optimizer, criterion, num_epochs=100)
