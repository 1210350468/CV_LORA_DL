# root/models/pretrain.py

import torch.nn as nn
from .backbone_ResNet import get_backbone
import torch
# 分类模型：Backbone加上分类头
class ClassificationModel(nn.Module):
    def __init__(self, backbone, num_classes=1000):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out

# 用于预训练和测试的通用函数
def get_classification_model(model_name, num_classes=1000, freeze_backbone=False):
    backbone = get_backbone(model_name)
    model = ClassificationModel(backbone, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model

# 加载预训练模型权重到指定模型
def load_pretrained_weights(model, pretrained_path):
    """
    将预训练权重加载到模型中。
    假设预训练模型保存时 Backbone 部分以 'backbone.' 开头。
    """
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = 'backbone.' + k if not k.startswith('backbone.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)