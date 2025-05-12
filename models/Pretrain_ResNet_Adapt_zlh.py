import torch
import torch.nn as nn
import torch.optim as optim
from .backbone_ResNet import get_backbone
from tqdm import tqdm
import numpy as np
import traceback
from .TripleBranchModule import TripleBranchModule
import torch.nn.functional as F
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
        self.fc = fc_block(backbone.output_dim, num_classes)
        self.last_loss = None
        self.last_acc = None
        self.logger = logger
        self.config = config
        self.n_way = config.n_way
        self.n_query = config.n_query
        self.n_support = config.n_support
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x, is_train=True):
        features = self.backbone(x)
        return self.fc(features, is_train)
    def fine_tune(self, support_images, lora_images, query_images, num_epochs=100, callback=None, batch_size=5, use_minibatch=True, sample_type=2):
        try:
            # 数据准备
            support = support_images.reshape(-1, 3, 224, 224).cuda()
            lora = lora_images.reshape(-1, 3, 224, 224).cuda()
            query = query_images.reshape(-1, 3, 224, 224).cuda()
            
            support_size = support.size(0)
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
            
            optimizer = optim.Adam([
                {'params': self.backbone.parameters(), 'lr': 1e-5},
                {'params': self.fc.parameters(), 'lr': 1e-4},
            ])

            self.backbone.train()
            for epoch in range(num_epochs):
                total_loss = 0.0
                rand_id = np.random.permutation(support_size)
                if use_minibatch:
                    # MiniBatch 训练模式
                    n_batches = support_size // batch_size
                    for i in range(n_batches):
                        optimizer.zero_grad()
                        # 设置两个minibatch采样方式，1：分层采样，2：随机采样
                        # 分层采样
                        if sample_type == 1:
                            batch_indices = []
                            for class_idx in range(self.n_way):
                                class_mask = (y_support == class_idx)
                                class_indices = torch.where(class_mask)[0]
                                selected = class_indices[torch.randperm(len(class_indices))[:batch_size//self.n_way]]
                                batch_indices.extend(selected.tolist())
                            
                            batch_indices = torch.tensor(batch_indices).cuda()
                            batch_support = support[batch_indices]
                            batch_labels = y_support[batch_indices]
                        # 随机采样
                        elif sample_type == 2:
                            selected_id =torch.from_numpy(rand_id[i:min(i+batch_size,support_size)]).cuda()
                            batch_support = support[selected_id]
                            batch_labels = y_support[selected_id]
                        
                        support_features = self.backbone(batch_support)
                        scores = self.fc(support_features)
                        loss = self.loss_fn(scores, batch_labels)
                        
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / n_batches
                else:
                    # 完整数据集训练模式
                    optimizer.zero_grad()
                    support_features = self.backbone(support)
                    scores = self.fc(support_features)
                    loss = self.loss_fn(scores, y_support)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss = loss.item()
                    avg_loss = total_loss
                
                if epoch % 10 == 0:
                    self.logger.log_workflow(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Mode: {'MiniBatch' if use_minibatch else 'Full Batch'}")
                
            # 测试阶段 - 分别记录train和eval模式的结果
            self.logger.log_workflow("\nStarting evaluation in train mode...")
            with torch.no_grad():
                self.backbone.train()
                # 记录train模式下的特征统计
                support_features_train = self.backbone(support)
                query_features_train = self.backbone(query)
                
                self.logger.log_workflow(f"""
                Train Mode Feature Statistics:
                Support Features - Mean: {support_features_train.mean().item():.4f}, Std: {support_features_train.std().item():.4f}
                Query Features - Mean: {query_features_train.mean().item():.4f}, Std: {query_features_train.std().item():.4f}
                """)
                
                # train模式下的分类结果
                support_scores_train = self.fc(support_features_train)
                query_scores_train = self.fc(query_features_train)
                
                support_acc_train = (support_scores_train.argmax(dim=1) == y_support).float().mean().item()
                query_acc_train = (query_scores_train.argmax(dim=1) == y_query).float().mean().item()
                
                # 记录分类分数的分布
                self.logger.log_workflow(f"""
                Train Mode Classification Statistics:
                Support Scores - Mean: {support_scores_train.mean().item():.4f}, Std: {support_scores_train.std().item():.4f}
                Query Scores - Mean: {query_scores_train.mean().item():.4f}, Std: {query_scores_train.std().item():.4f}
                Support Accuracy: {support_acc_train*100:.2f}%
                Query Accuracy: {query_acc_train*100:.2f}%
                """)
                
                # 切换到eval模式测试
                self.logger.log_workflow("\nStarting evaluation in eval mode...")
                self.backbone.eval()
                # 记录eval模式下的特征统计
                support_features_eval = self.backbone(support)
                query_features_eval = self.backbone(query)
                
                self.logger.log_workflow(f"""
                Eval Mode Feature Statistics:
                Support Features - Mean: {support_features_eval.mean().item():.4f}, Std: {support_features_eval.std().item():.4f}
                Query Features - Mean: {query_features_eval.mean().item():.4f}, Std: {query_features_eval.std().item():.4f}
                Feature Difference (Train-Eval):
                Support - Mean Diff: {(support_features_train - support_features_eval).abs().mean().item():.4f}
                Query - Mean Diff: {(query_features_train - query_features_eval).abs().mean().item():.4f}
                """)
                
                # eval模式下的分类结果
                support_scores_eval = self.fc(support_features_eval)
                query_scores_eval = self.fc(query_features_eval)
                
                support_acc_eval = (support_scores_eval.argmax(dim=1) == y_support).float().mean().item()
                query_acc_eval = (query_scores_eval.argmax(dim=1) == y_query).float().mean().item()
                
                # 记录分类分数的分布
                self.logger.log_workflow(f"""
                Eval Mode Classification Statistics:
                Support Scores - Mean: {support_scores_eval.mean().item():.4f}, Std: {support_scores_eval.std().item():.4f}
                Query Scores - Mean: {query_scores_eval.mean().item():.4f}, Std: {query_scores_eval.std().item():.4f}
                Support Accuracy: {support_acc_eval*100:.2f}%
                Query Accuracy: {query_acc_eval*100:.2f}%
                """)
                
                # 记录预测标签的分布
                pred_train = query_scores_train.argmax(dim=1)
                pred_eval = query_scores_eval.argmax(dim=1)
                
                self.logger.log_workflow(f"""
                Prediction Distribution:
                Train Mode - Unique predictions: {torch.unique(pred_train, return_counts=True)}
                Eval Mode - Unique predictions: {torch.unique(pred_eval, return_counts=True)}
                """)
            
            return query_acc_train  # 返回train模式下的准确率
            
        except Exception as e:
            self.logger.log_workflow_error(f"Error in fine_tune: {str(e)}")
            self.logger.log_workflow_error(f"Error traceback: {traceback.format_exc()}")
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
def get_test_model(model_name, num_classes=1000, freeze_backbone=False,logger=None,config=None):
    backbone = get_backbone(model_name)
    model = FineTuningModel(backbone, num_classes,logger,config)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
