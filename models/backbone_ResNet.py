# root/models/backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# BasicBlock: 用于ResNet18和ResNet34的残差块
class BasicBlock(nn.Module):
    expansion = 1  # 用于控制输出维度的扩展

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层，使用3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample  # 捷径连接的下采样层
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入作为捷径连接的输入

        # 第一个卷积层、批归一化和ReLU激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 第二个卷积层、批归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样层，则应用下采样到捷径连接
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = F.relu(out)

        return out

# BottleneckBlock: 用于ResNet50及更深层次的网络
class BottleneckBlock(nn.Module):
    expansion = 4  # Bottleneck block将输出通道扩展为4倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        
        # 1x1卷积用于降低维度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积处理特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1卷积用于恢复维度
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入作为捷径连接的输入

        # 1x1卷积降低维度
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 3x3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # 1x1卷积恢复维度
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果有下采样层，则应用下采样到捷径连接
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = F.relu(out)

        return out

# ResNet类
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始化 self.in_channels
        # 初始卷积层和批归一化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建每一层残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # 如果输入和输出维度不匹配，进行下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 堆叠残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积、批归一化和ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 池化和全连接层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 工厂函数，用于创建不同的 ResNet 变体
def create_resnet(model_name, num_classes=1000):
    model_dict = {
        'ResNet18': (BasicBlock, [2, 2, 2, 2]),
        'ResNet34': (BasicBlock, [3, 4, 6, 3]),
        'ResNet50': (BottleneckBlock, [3, 4, 6, 3]),
        'ResNet101': (BottleneckBlock, [3, 4, 23, 3]),
        'ResNet152': (BottleneckBlock, [3, 8, 36, 3]),
    }
    if model_name not in model_dict:
        raise ValueError(f"Unsupported model name {model_name}. Supported models are: {list(model_dict.keys())}")
    block, layers = model_dict[model_name]
    return ResNet(block, layers, num_classes=num_classes)

# Backbone模型：基于ResNet架构的特征提取器
class Backbone(nn.Module):
    def __init__(self, model_name):
        super(Backbone, self).__init__()
        # 使用工厂函数创建 ResNet 模型
        resnet = create_resnet(model_name)
        # 直接包含 ResNet 的各个层，而不是使用 nn.Sequential 包装
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.output_dim = 512 * resnet.layer4[0].expansion  # 512 * expansion

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
# 工厂函数，用于创建 Backbone
def get_backbone(model_name):
    return Backbone(model_name)
