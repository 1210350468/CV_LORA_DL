import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度训练模块

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 导入模型和数据加载器
from dataloader.Pretrain_Dataloader import get_pretrain_dataloader
from dataloader.Pretrain_Dataloader import mixup_data, mixup_criterion
import torchvision.models as models

# 训练函数
def train_model(model, train_loader, train_loader_aug, optimizer_Adam, scheduler_Adam, criterion, num_epochs=10, save_path='../output/pretrain/resnet', mixup_alpha=1.0, accumulation_steps=4):
    # 如果保存路径不存在，创建路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = model.cuda()  # 确保模型在 GPU 上
    scaler = GradScaler()  # 初始化梯度缩放器用于混合精度训练

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        flag = False
        mixup = False

        # 选择数据加载器和优化器
        if flag:
            if epoch + 1 <= int(num_epochs / 2):
                data_loader = train_loader_aug
                optimizer = optimizer_Adam
                scheduler = scheduler_Adam
                print(f"Epoch {epoch+1}: 使用 train_loader_aug")
            else:
                data_loader = train_loader
                print(f"Epoch {epoch+1}: 使用 train_loader")
        else:
            data_loader = train_loader_aug
            optimizer = optimizer_Adam
            print(f"Epoch {epoch+1}: 使用 train_loader")
            print(f"Epoch {epoch+1}: 使用 optimizer_Adam")

        # 使用 tqdm 显示进度条
        for i, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.cuda(), labels.cuda()

            # 应用 mixup
            if epoch + 1 > int(num_epochs * 3 / 4):
                mixup = True
            else:
                mixup = False

            if mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)

            optimizer.zero_grad()

            # 自动混合精度
            with autocast():
                outputs = model(inputs)
                if mixup:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, labels)

            # 梯度累积
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()*accumulation_steps

        # scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # 每 50 个 epoch 保存一次模型
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'{save_path}model_{epoch+1}.pth')
            print(f"Model saved at {save_path}model_{epoch+1}.pth")

        # 每 10 个 epoch 进行一次验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            print("Testing...")
            with torch.no_grad():
                acc = []
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss
                    # 获取验证集上的准确率
                    acc.append((outputs.argmax(dim=1) == labels).float().mean().item())

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Acc: {sum(acc) / len(acc) * 100} % ")
                with open(f'{save_path}log.txt', 'a') as f:
                    f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Acc: {sum(acc) / len(acc) * 100} % \n")

if __name__ == "__main__":
    # 选择 ResNet 模型
    # 获取预训练的 ResNet-18 模型
    model = models.resnet18(pretrained=True)

    # 修改最后的全连接层以适应你的任务
    num_classes = 100 
    torch.save(model.state_dict(), 'resnet18_im1k.pth')
    
    input("Press Enter to continue...")
    model = model.cuda()
    # 设置优化器和损失函数
    optimizer_Adam = optim.Adam(model.parameters(), lr=3e-4)
    scheduler_Adam = CosineAnnealingLR(optimizer_Adam, T_max=50, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    # 准备数据
    train_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImage_train.json')
    train_loader_aug = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImage_train.json', aug=True)
    val_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImage_val.json')

    # 启动训练
    # train_model(model, train_loader, train_loader_aug,optimizer_Adam, scheduler_Adam, criterion, num_epochs=100)
