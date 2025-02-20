# root/models/prototypenet.py

import torch
import torch.nn as nn
from .backbone_ResNet import get_backbone

# ProtoNet模型：包含元学习的前向逻辑，并内部构建标签
class ProtoNet(nn.Module):
    def __init__(self, backbone, N_way=5, K_shot=1, Q_query=15):
        super(ProtoNet, self).__init__()
        self.backbone = backbone
        self.N_way = N_way
        self.K_shot = K_shot
        self.Q_query = Q_query

    def forward(self, support_images, query_images):
        """
        Args:
            support_images: [N_way * K_shot, C, H, W]
            query_images: [N_way * Q_query, C, H, W]
        Returns:
            distances: [N_way * Q_query, N_way]
            query_labels: [N_way * Q_query]
        """
        device = support_images.device

        # 提取特征
        support_features = self.backbone(support_images)  # [N_way * K_shot, D]
        query_features = self.backbone(query_images)      # [N_way * Q_query, D]

        # 构建标签
        support_labels = torch.arange(self.N_way).unsqueeze(1).repeat(1, self.K_shot).view(-1).to(device)  # [N_way * K_shot]
        query_labels = torch.arange(self.N_way).unsqueeze(1).repeat(1, self.Q_query).view(-1).to(device)      # [N_way * Q_query]

        # 计算每个类别的原型
        prototypes = []
        for c in range(self.N_way):
            class_features = support_features[support_labels == c]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # [N_way, D]

        # 计算查询样本到原型的距离
        distances = self.euclidean_distance(query_features, prototypes)  # [N_way * Q_query, N_way]

        return distances, query_labels

    @staticmethod
    def euclidean_distance(a, b):
        """
        计算欧氏距离
        a: [N, D]
        b: [M, D]
        返回: [N, M]
        """
        n = a.size(0)
        m = b.size(0)
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.pow(a - b, 2).sum(2)

# 工厂函数，用于创建 ProtoNet
def get_protonet(model_name, N_way=5, K_shot=5, Q_query=15):
    backbone = get_backbone(model_name)
    return ProtoNet(backbone, N_way, K_shot, Q_query)
