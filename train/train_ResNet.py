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
def train_model(model, train_loader, train_loader_aug, optimizer_Adam, scheduler_Adam, criterion, num_epochs=10, save_path='../output/pretrain/resnet', mixup_alpha=1.0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = model.cuda()
    best_acc = 0
    best_state = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        
        # 前一半epoch使用增强数据，后一半使用原始数据
        data_loader = train_loader_aug if epoch < num_epochs // 2 else train_loader
        print(f"Epoch {epoch+1}: Using {'augmented' if epoch < num_epochs // 2 else 'original'} data")

        # 训练阶段
        for i, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # 在后期使用mixup
            if epoch + 1 > int(num_epochs * 3 / 4):
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
                
            optimizer_Adam.zero_grad()
            outputs = model(inputs)
            
            # 计算损失
            if epoch + 1 > int(num_epochs * 3 / 4):
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer_Adam.step()
            running_loss += loss.item()

        # 更新学习率
        scheduler_Adam.step()
        
        # 计算当前epoch的平均损失
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # 每10个epoch验证一次
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            val_acc = []
            
            print("Testing...")
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc.append((outputs.argmax(dim=1) == labels).float().mean().item())
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = sum(val_acc) / len(val_acc) * 100
                
                print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.2f}%")
                
                # 保存日志
                with open(f'{save_path}log.txt', 'a') as f:
                    f.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%\n")
                
                # 保存最佳模型
                if avg_val_acc > best_acc:
                    best_acc = avg_val_acc
                    best_state = model.state_dict()
                    torch.save(best_state, f'{save_path}/best_model.pth')
                    print(f"New best model saved! Accuracy: {best_acc:.2f}%")
        
        # 每50个epoch保存checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_Adam.state_dict(),
                'scheduler_state_dict': scheduler_Adam.state_dict(),
                'best_acc': best_acc,
            }, f'{save_path}/checkpoint_epoch{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")

    # 训练结束，加载并返回最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

if __name__ == "__main__":
    # 选择 ResNet 模型
    # 获取预训练的 ResNet-18 模型
    model = models.resnet18(pretrained=False)

    # 修改最后的全连接层以适应你的任务
    num_classes = 100 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    torch.save(model.state_dict(), 'resnet18_im1k11.pth')
    
    input("Press Enter to continue...")
    model = model.cuda()
    # 设置优化器和损失函数
    optimizer_Adam = optim.Adam(model.parameters(), lr=3e-4)
    scheduler_Adam = CosineAnnealingLR(optimizer_Adam, T_max=50, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    # 准备数据
    train_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImagenet_train.json')
    train_loader_aug = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImagenet_train.json', aug=True)
    val_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImagenet_val.json')

    # 启动训练
    train_model(model, train_loader, train_loader_aug,optimizer_Adam, scheduler_Adam, criterion, num_epochs=500)
