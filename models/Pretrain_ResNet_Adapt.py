# root/models/test.py

import torch
import torch.nn as nn
import torch.optim as optim
from .Pretrain_ResNet import ClassificationModel, load_pretrained_weights
from .backbone_ResNet import get_backbone
from tqdm import tqdm
import numpy as np
import traceback
# 微调模型：Backbone加上分类头，并包含微调逻辑
class fc_block(nn.Module):
    #要有softmax
    def __init__(self, in_dim, out_dim):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
class FineTuningModel(nn.Module):
    def __init__(self, backbone, num_classes=1000):
        super(FineTuningModel, self).__init__()
        self.backbone = backbone
        self.fc = fc_block(backbone.output_dim, num_classes)
        self.last_loss = None
        self.last_acc = None

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out

    def fine_tune(self, support_images, query_images, num_epochs=100, callback=None, batch_size=5):
        try:
            self.cuda()
            self.train()
            
            n_way = support_images.shape[0]
            k_shot = support_images.shape[1]
            q_query = query_images.shape[1]
            
            support = support_images.contiguous()
            support = support.view(n_way*k_shot, *support.size()[2:]).cuda()
            query = query_images.contiguous()
            query = query.view(n_way*q_query, *query.size()[2:]).cuda()
            
            optimizer = optim.Adam([
                {'params': self.backbone.parameters(), 'lr': 3e-5},
                {'params': self.fc.parameters(), 'lr': 3e-4}
            ])
            
            loss_fn = nn.CrossEntropyLoss()
            y_support = torch.from_numpy(np.repeat(range(n_way), k_shot)).cuda()
            y_query = torch.from_numpy(np.repeat(range(n_way), q_query)).cuda()
            
            support_size = support.size(0)
            
            for epoch in range(num_epochs):
                rand_id = np.random.permutation(support_size)
                epoch_loss = 0.0
                
                for i in range(0, support_size, batch_size):
                    optimizer.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                    z_batch = support[selected_id]
                    y_batch = y_support[selected_id]
                    
                    scores = self(z_batch)
                    loss = loss_fn(scores, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # 计算当前epoch的平均loss
                avg_loss = epoch_loss / (support_size / batch_size)
                self.last_loss = avg_loss
                
                # 计算当前准确率
                with torch.no_grad():
                    scores = self(query)
                    acc = (scores.argmax(dim=1) == y_query).float().mean().item()
                    self.last_acc = acc
                
                if callback:
                    callback(epoch + 1, num_epochs, avg_loss, acc)
                
            return self.last_acc
            
        except Exception as e:
            print(f"Error in fine_tune: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            raise
# 加载预训练模型权重到指定模型
def load_pretrained_weights(model, pretrained_path):
    """
    将预训练权重加载到模型中。
    假设预训练模型保存时 Backbone 部分以 'backbone.' 开头。
    """
    
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k] = v
        else:
            # new_state_dict[k] = v
            continue
    model.load_state_dict(new_state_dict, strict=False)
# 获取测试模型实例
def get_test_model(model_name, num_classes=1000, freeze_backbone=False):
    backbone = get_backbone(model_name)
    model = FineTuningModel(backbone, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
