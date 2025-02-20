#构建一个测试预训练模型效果的脚本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import os
import torchvision.models as models
# 添加模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader.Pretrain_Dataloader import get_pretrain_dataloader
from models.Pretrain_ResNet import get_classification_model, load_pretrained_weights
from models.Pretrain_VIT import get_vit_model


def test_model(model, test_loader):
    model = model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total} %")
# 测试模型
if __name__ == '__main__':
    # 准备数据
    save_path = '../output/pretrain/ResNet18'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_loader = get_pretrain_dataloader(batch_size=64, json_file='../utils/miniImage_val.json')
    # model=resnet18(num_classes=100)
    model=get_classification_model(model_name='resnet18', num_classes=100)
    # model=get_resnet_model(version='resnet18', num_classes=100)
    model = model.cuda()
    # 加载预训练模型
    # model.load_state_dict(torch.load('../output/pretrain/model_400.pth'))
    # load_pretrained_weights(model, '../output/pretrain/model_100.pth')
    load_pretrained_weights(model, '../output/pretrain/ResNet18/resnet18_im1k.pth')
    torch.save(model.state_dict(), f'{save_path}/model_im1k.pth')
    #打印key
    print(model.state_dict().keys())

    # 测试函数
    test_model(model, test_loader)
