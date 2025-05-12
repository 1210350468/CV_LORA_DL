import torch
import torch.nn as nn
import torch.optim as optim
from .backbone_ResNet import get_backbone
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import traceback
from .TripleBranchModule import TripleBranchModule
import torch.nn.functional as F
from .backbone_ResNet10 import ResNet10
class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x
# 余弦分类器
class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, temperature=0.07):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.temperature = temperature
    def forward(self, query_feat, support_feat, support_labels):
        prototypes = []
        unique_labels = torch.unique(support_labels)
        for lbl in unique_labels:
            mask = (support_labels == lbl)
            class_features = support_feat[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # [N, D]
        query_feat = F.normalize(query_feat, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        sim = torch.mm(query_feat, prototypes.t()) / self.temperature
        return self.scale * sim
class fc_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class FineTuningModel(nn.Module):
    def __init__(self, backbone, num_classes=1000,logger=None,config=None):
        super(FineTuningModel, self).__init__()
        self.backbone = backbone
        self.loss_fn = nn.CrossEntropyLoss()
        if config.model_name == 'ResNet10':
            self.fc = fc_block(backbone.final_feat_dim, num_classes)
            self.triple_branch = TripleBranchModule(backbone.final_feat_dim, self.loss_fn)
        else:
            self.fc = fc_block(backbone.output_dim, num_classes)
            self.triple_branch = TripleBranchModule(backbone.output_dim, self.loss_fn)
        self.last_loss = None
        self.last_acc = None
        self.logger = logger
        self.config = config
        self.n_way = config.n_way
        self.n_query = config.n_query
        self.n_support = config.n_support

        self.freeze_backbone = False
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)
    def fine_tune(self, support_images, lora_images, query_images, num_epochs=100, callback=None, batch_size=5, use_minibatch=True, sample_type=2):
        # 数据准备
        support = support_images.reshape(-1, 3, 224, 224).cuda()
        lora = lora_images.reshape(-1, 3, 224, 224).cuda()
        query = query_images.reshape(-1, 3, 224, 224).cuda()
        np.random.seed(10)

        support_size = support.size(0)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_lora = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        if not self.freeze_backbone:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.backbone.parameters()), lr = 0.01)
            
        
        # 设置训练模式 - 修复点1：在训练前明确设置模型模式
        if not self.freeze_backbone:
            self.backbone.train()
        else:
            self.backbone.eval()
        for epoch in range(num_epochs):
            total_loss = 0.0
            rand_id = np.random.permutation(support_size)
            if use_minibatch:
                # MiniBatch 训练模式
                for j in range(0, support_size, batch_size):
                    if not self.freeze_backbone:
                        delta_opt.zero_grad()   
                    # 修复点2：使用与标准finetune.py相同的batch采样逻辑
                    selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
                    batch_support = support[selected_id]
                    batch_labels = y_support[selected_id]
                    batch_lora = lora[selected_id]
                    batch_lora_labels = y_lora[selected_id]
                    support_features = self.backbone(batch_support)
                    lora_features = self.backbone(batch_lora)
                    # combined_features = torch.cat([support_features, lora_features], dim=0)
                    # combined_labels = torch.cat([batch_labels, batch_lora_labels], dim=0)
                    # scores = self.fc(combined_features)
                    # loss = self.loss_fn(scores, combined_labels)
                    ft_support,ft_lora,ft_mixup = self.triple_branch(support_features,lora_features,epoch,self.logger)
                    scores_support = self.fc(ft_support)
                    scores_lora = self.fc(ft_lora)
                    scores_mixup = self.fc(ft_mixup)
                    loss_support = self.loss_fn(scores_support, batch_labels)
                    loss_lora = self.loss_fn(scores_lora, batch_lora_labels)
                    loss_mixup = self.loss_fn(scores_mixup, batch_lora_labels)
                    loss = loss_support + loss_lora + loss_mixup
                    loss.backward()
                    # classifier_opt.step()
                    if not self.freeze_backbone:
                        delta_opt.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / (support_size // batch_size + (0 if support_size % batch_size == 0 else 1))
            else:
                # 完整数据集训练模式
                # classifier_opt.zero_grad()
                if not self.freeze_backbone:
                    delta_opt.zero_grad()
                support_features = self.backbone(support)
                scores = self.fc(support_features)
                loss = self.loss_fn(scores, y_support)
                
                loss.backward()
                # classifier_opt.step()
                if not self.freeze_backbone:
                    delta_opt.step()
                total_loss = loss.item()
                avg_loss = total_loss
            
            if epoch % 10 == 0:
                self.logger.log_workflow(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Mode: {'MiniBatch' if use_minibatch else 'Full Batch'}")
            
        # 测试阶段 - 分别记录train和eval模式的结果
        self.logger.log_workflow("\nStarting evaluation in train mode...")
        
        # 修复点3：设置为eval模式时，确保所有组件都进入eval模式
        self.backbone.eval()
        # classifier.eval()
        
        with torch.no_grad():  # 修复点4：添加torch.no_grad()避免梯度计算
            query_features = self.backbone(query)
            query_scores = self.fc(query_features)
            
            y_query_cal = np.repeat(range(self.n_way), self.n_query)
            print("y_query:" ,y_query,"y_query_cal:" ,y_query_cal)
            topk_scores, topk_labels = query_scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query_cal)
            correct_this, count_this = float(top1_correct), len(y_query_cal)
            query_acc_train = float(correct_this)/ float(count_this) *100
            print("query_acc_train:" ,query_acc_train)
            
        return query_acc_train  # 返回的准确率
# 加载预训练模型权重到指定模型
def load_pretrained_weights(model, pretrained_path,File_format='pth'):
    """
    将预训练权重加载到模型中。
    预训练模型保存时 Backbone 部分以 'backbone.' 开头。
    """
    if File_format == 'pth':
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    elif File_format == 'tar':
        tmp = torch.load(pretrained_path)
        state = tmp['state']
        state_keys = list(state.keys())
        # print("state_keys:",state_keys)
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","backbone.")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
        model.load_state_dict(state, strict=False)
# 获取测试模型实例
def get_test_model(model_name, num_classes=1000, freeze_backbone=False,logger=None,config=None):
    if model_name == 'ResNet10':
        backbone = ResNet10()
    else:
        backbone = get_backbone(model_name)
    model = FineTuningModel(backbone, num_classes,logger,config)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.freeze_backbone = True
    else:
        model.freeze_backbone = False

    return model
