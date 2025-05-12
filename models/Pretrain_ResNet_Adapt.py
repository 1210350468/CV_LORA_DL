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
        if config.model_name == 'ResNet10':
            self.fc = fc_block(backbone.final_feat_dim, num_classes)
        else:
            self.fc = fc_block(backbone.output_dim, num_classes)
        self.last_loss = None
        self.last_acc = None
        self.logger = logger
        self.config = config
        self.n_way = config.n_way
        self.n_support = config.n_support
        self.n_query = config.n_query
        self.loss_fn = nn.CrossEntropyLoss()
        self.freeze_backbone = config.freeze_backbone
        print("config:",self.freeze_backbone)
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)
    
    def fine_tune(self, support_images, lora_images, query_images, num_epochs=100, callback=None, batch_size=5, use_minibatch=True, sample_type=2):
        """
        渐进式融合策略的微调方法：
        1. 第一阶段：仅用支持集训练建立基础
        2. 第二阶段：混合训练，更多关注支持集
        3. 第三阶段：知识蒸馏训练，让LoRA图像特征向支持集特征靠拢
        """
        # 数据准备
        support = support_images.reshape(-1, 3, 224, 224).cuda()
        lora = lora_images.reshape(-1, 3, 224, 224).cuda()
        query = query_images.reshape(-1, 3, 224, 224).cuda()
        np.random.seed(10)

        support_images = support_images.cuda()
        lora_images = lora_images.cuda()
        query_images = query_images.cuda()
        
        # 准备标签
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_lora = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        
        # 初始化优化器
        if not self.freeze_backbone:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
        else:
            optimizer = torch.optim.SGD(self.fc.parameters(), lr=0.01)
            
        self.backbone.cuda()
        
        # 设置训练模式
        if not self.freeze_backbone:
            self.backbone.train()
        else:
            self.backbone.eval()
        
        # 阶段比例
        first_stage = int(num_epochs * 0.5)  # 50%: 仅支持集
        second_stage = int(num_epochs * 0.3) # 30%: 混合训练
        # 剩余20%: 知识蒸馏
        
        # 记录最佳模型
        best_acc = 0.0
        best_state = None
        
        # 第一阶段：仅使用支持集训练
        self.logger.log_workflow("\n===== 阶段1: 仅使用支持集训练 =====")
        for epoch in range(first_stage):
            # 在每个epoch随机打乱支持集
            rand_idx = torch.randperm(support.size(0))
            shuffled_support = support[rand_idx]
            shuffled_y_support = y_support[rand_idx]
            
            if use_minibatch:
                # Mini-batch 训练
                total_loss = 0.0
                for j in range(0, support.size(0), batch_size):
                    end_idx = min(j + batch_size, support.size(0))
                    batch_support = shuffled_support[j:end_idx]
                    batch_y_support = shuffled_y_support[j:end_idx]
                    
                    optimizer.zero_grad()
                    features = self.backbone(batch_support)
                    scores = self.fc(features)
                    loss = self.loss_fn(scores, batch_y_support)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / ((support.size(0) + batch_size - 1) // batch_size)
            else:
                # 完整批次训练
                optimizer.zero_grad()
                features = self.backbone(support)
                scores = self.fc(features)
                loss = self.loss_fn(scores, y_support)
                loss.backward()
                optimizer.step()
                avg_loss = loss.item()
                
            if epoch % 10 == 0:
                self.logger.log_workflow(f"阶段1 - Epoch {epoch}/{first_stage}: Loss = {avg_loss:.4f}")
                
                # 每10个epoch检查一次在查询集上的性能
                if epoch > 0 and epoch % 30 == 0:
                    self.backbone.eval()
                    with torch.no_grad():
                        query_features = self.backbone(query)
                        query_scores = self.fc(query_features)
                        preds = torch.argmax(query_scores, dim=1)
                        acc = (preds == y_query).float().mean().item() * 100
                        self.logger.log_workflow(f"  查询集准确率: {acc:.2f}%")
                        
                        # 保存最佳模型
                        if acc > best_acc:
                            best_acc = acc
                            best_state = {
                                'backbone': self.backbone.state_dict(),
                                'fc': self.fc.state_dict()
                            }
                    
                    if not self.freeze_backbone:
                        self.backbone.train()
        
        # 第二阶段：渐进式混合训练
        self.logger.log_workflow("\n===== 阶段2: 混合训练(偏向支持集) =====")
        start_epoch = first_stage
        end_epoch = first_stage + second_stage
        
        for epoch in range(start_epoch, end_epoch):
            total_loss = 0.0
            support_loss_sum = 0.0
            lora_loss_sum = 0.0
            
            # 计算当前LoRA样本权重，随着epoch增加而增加
            progress = (epoch - start_epoch) / second_stage  # 0->1
            lora_weight = 0.2 + 0.3 * progress  # 0.2->0.5
            
            # 创建混合批次的索引
            support_indices = []
            lora_indices = []
            
            # 确定本epoch使用多少LoRA样本
            lora_sample_count = int(support.size(0) * lora_weight)
            if lora_sample_count > 0:
                # 随机选择LoRA样本
                lora_indices = np.random.choice(
                    range(lora.size(0)), 
                    size=lora_sample_count,
                    replace=False
                )
            
            # 使用所有支持集样本
            support_indices = np.arange(support.size(0))
            
            # 合并并打乱索引
            if use_minibatch:
                # Mini-batch训练
                for j in range(0, len(support_indices), batch_size):
                    end_idx = min(j + batch_size, len(support_indices))
                    batch_indices = support_indices[j:end_idx]
                    
                    optimizer.zero_grad()
                    batch_support = support[batch_indices]
                    batch_y_support = y_support[batch_indices]
                    
                    features = self.backbone(batch_support)
                    scores = self.fc(features)
                    loss = self.loss_fn(scores, batch_y_support)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    support_loss_sum += loss.item()
                
                # 训练LoRA样本(如果有)
                if len(lora_indices) > 0:
                    for j in range(0, len(lora_indices), batch_size):
                        end_idx = min(j + batch_size, len(lora_indices))
                        batch_indices = lora_indices[j:end_idx]
                        
                        optimizer.zero_grad()
                        batch_lora = lora[batch_indices]
                        batch_y_lora = y_lora[batch_indices]
                        
                        features = self.backbone(batch_lora)
                        scores = self.fc(features)
                        # 对LoRA样本损失赋予较低权重
                        loss = self.loss_fn(scores, batch_y_lora) * 0.7
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        lora_loss_sum += loss.item()
            else:
                # 完整批次训练
                # 支持集前向传播
                optimizer.zero_grad()
                features_support = self.backbone(support)
                scores_support = self.fc(features_support)
                loss_support = self.loss_fn(scores_support, y_support)
                
                # LoRA批次(如果有)
                if len(lora_indices) > 0:
                    batch_lora = lora[lora_indices]
                    batch_y_lora = y_lora[lora_indices]
                    features_lora = self.backbone(batch_lora)
                    scores_lora = self.fc(features_lora)
                    loss_lora = self.loss_fn(scores_lora, batch_y_lora) * 0.7
                    
                    # 合并损失
                    loss = loss_support + loss_lora
                    loss.backward()
                    optimizer.step()
                    
                    support_loss_sum = loss_support.item()
                    lora_loss_sum = loss_lora.item()
                    total_loss = loss.item()
                else:
                    # 只有支持集
                    loss = loss_support
                    loss.backward()
                    optimizer.step()
                    
                    support_loss_sum = loss_support.item()
                    total_loss = loss.item()
            
            if epoch % 10 == 0:
                lora_frac = len(lora_indices) / support.size(0)
                self.logger.log_workflow(
                    f"阶段2 - Epoch {epoch}/{end_epoch}: "
                    f"Total Loss = {total_loss:.4f}, "
                    f"Support Loss = {support_loss_sum:.4f}, "
                    f"Lora Loss = {lora_loss_sum:.4f}, "
                    f"Lora使用比例: {lora_frac:.2f}"
                )
                
                # 检查查询集性能
                if epoch % 30 == 0:
                    self.backbone.eval()
                    with torch.no_grad():
                        query_features = self.backbone(query)
                        query_scores = self.fc(query_features)
                        preds = torch.argmax(query_scores, dim=1)
                        acc = (preds == y_query).float().mean().item() * 100
                        self.logger.log_workflow(f"  查询集准确率: {acc:.2f}%")
                        
                        # 保存最佳模型
                        if acc > best_acc:
                            best_acc = acc
                            best_state = {
                                'backbone': self.backbone.state_dict(),
                                'fc': self.fc.state_dict()
                            }
                    
                if not self.freeze_backbone:
                        self.backbone.train()
        
        # 第三阶段：知识蒸馏
        self.logger.log_workflow("\n===== 阶段3: 知识蒸馏训练 =====")
        start_epoch = end_epoch
        end_epoch = num_epochs
        
        # 在这个阶段，我们希望LoRA生成的图像特征与支持集特征对齐
        for epoch in range(start_epoch, end_epoch):
            # 计算支持集特征(不更新梯度)
            with torch.no_grad():
                support_features = self.backbone(support)
            
            # 计算LoRA图像特征并蒸馏
            optimizer.zero_grad()
            lora_features = self.backbone(lora)
            
            # 1. 蒸馏损失：让LoRA特征接近支持集特征
            distill_loss = F.mse_loss(lora_features, support_features)
            
            # 2. LoRA图像的分类损失
            lora_scores = self.fc(lora_features)
            classif_loss = self.loss_fn(lora_scores, y_lora)
            
            # 综合损失
            loss = classif_loss + 0.5 * distill_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                self.logger.log_workflow(
                    f"阶段3 - Epoch {epoch}/{end_epoch}: "
                    f"总损失 = {loss.item():.4f}, "
                    f"分类损失 = {classif_loss.item():.4f}, "
                    f"蒸馏损失 = {distill_loss.item():.4f}"
                )
                
                # 检查查询集性能
                if epoch % 30 == 0:
                    self.backbone.eval()
                    with torch.no_grad():
                        query_features = self.backbone(query)
                        query_scores = self.fc(query_features)
                        preds = torch.argmax(query_scores, dim=1)
                        acc = (preds == y_query).float().mean().item() * 100
                        self.logger.log_workflow(f"  查询集准确率: {acc:.2f}%")
                        
                        # 保存最佳模型
                        if acc > best_acc:
                            best_acc = acc
                            best_state = {
                                'backbone': self.backbone.state_dict(),
                                'fc': self.fc.state_dict()
                            }
                    
                    if not self.freeze_backbone:
                        self.backbone.train()
        
        # 最终评估前，恢复最佳模型
        if best_state is not None:
            self.backbone.load_state_dict(best_state['backbone'])
            self.fc.load_state_dict(best_state['fc'])
        
        # 最终评估
        self.logger.log_workflow("\n===== 最终评估 =====")
        self.backbone.eval()
        
        with torch.no_grad():
            query_features = self.backbone(query)
            query_scores = self.fc(query_features)
            
            y_query_cal = np.repeat(range(self.n_way), self.n_query)
            topk_scores, topk_labels = query_scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query_cal)
            correct_this, count_this = float(top1_correct), len(y_query_cal)
            query_acc = float(correct_this) / float(count_this) * 100
            
            self.logger.log_workflow(f"最终查询集准确率: {query_acc:.2f}%")
            if best_acc > 0:
                self.logger.log_workflow(f"最佳查询集准确率: {best_acc:.2f}%")
            
        return max(query_acc, best_acc)

    def lora_only_fine_tune(self, support_images, lora_images, query_images, num_epochs=100, callback=None, batch_size=5, use_minibatch=True, sample_type=2):
        """
        仅使用LoRA图像进行微调
        """
        # 数据准备
        support = support_images.reshape(-1, 3, 224, 224).cuda()
        lora = lora_images.reshape(-1, 3, 224, 224).cuda()
        query = query_images.reshape(-1, 3, 224, 224).cuda()
        np.random.seed(10)
        
        # 准备标签
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_lora = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        print("self.freeze_backbone:",self.freeze_backbone)
        # 初始化优化器
        if not self.freeze_backbone:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
            self.logger.log_workflow("backbone is not frozen")
        else:
            optimizer = torch.optim.SGD(self.fc.parameters(), lr=0.01)
            self.logger.log_workflow("backbone is frozen")
        self.backbone.cuda()
        
        # 设置训练模式
        if not self.freeze_backbone:
            self.backbone.train()
        else:
            self.backbone.eval()
        
        # 记录最佳模型
        best_acc = 0.0
        best_state = None
        
        self.logger.log_workflow("\n===== 仅使用LoRA图像进行训练 =====")
        for epoch in range(num_epochs):
            # 在每个epoch随机打乱LoRA图像
            rand_idx = torch.randperm(lora.size(0))
            shuffled_lora = lora[rand_idx]
            shuffled_y_lora = y_lora[rand_idx]
            
            if use_minibatch:
                # Mini-batch训练
                total_loss = 0.0
                for j in range(0, lora.size(0), batch_size):
                    end_idx = min(j + batch_size, lora.size(0))
                    batch_lora = shuffled_lora[j:end_idx]
                    batch_y_lora = shuffled_y_lora[j:end_idx]
                    
                    optimizer.zero_grad()
                    features = self.backbone(batch_lora)
                    scores = self.fc(features)
                    loss = self.loss_fn(scores, batch_y_lora)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / ((lora.size(0) + batch_size - 1) // batch_size)
            else:
                # 完整批次训练
                optimizer.zero_grad()
                features = self.backbone(lora)
                scores = self.fc(features)
                loss = self.loss_fn(scores, y_lora)
                loss.backward()
                optimizer.step()
                avg_loss = loss.item()
            
            if epoch % 10 == 0:
                self.logger.log_workflow(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
                # 每30个epoch评估一次
                if epoch > 0 and epoch % 30 == 0:
                    self.backbone.eval()
                    with torch.no_grad():
                        # 在查询集上评估
                        query_features = self.backbone(query)
                        query_scores = self.fc(query_features)
                        preds = torch.argmax(query_scores, dim=1)
                        acc = (preds == y_query).float().mean().item() * 100
                        self.logger.log_workflow(f"  查询集准确率: {acc:.2f}%")
                        
                        # 保存最佳模型
                        if acc > best_acc:
                            best_acc = acc
                            best_state = {
                                'backbone': self.backbone.state_dict(),
                                'fc': self.fc.state_dict()
                            }
                    
                    if not self.freeze_backbone:
                        self.backbone.train()
        
        # 恢复最佳模型
        if best_state is not None:
            self.backbone.load_state_dict(best_state['backbone'])
            self.fc.load_state_dict(best_state['fc'])
        
        # 最终评估
        self.backbone.eval()
        with torch.no_grad():
            query_features = self.backbone(query)
            query_scores = self.fc(query_features)
            
            y_query_cal = np.repeat(range(self.n_way), self.n_query)
            topk_scores, topk_labels = query_scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query_cal)
            correct_this, count_this = float(top1_correct), len(y_query_cal)
            query_acc = float(correct_this) / float(count_this) * 100
            
            self.logger.log_workflow(f"最终查询集准确率: {query_acc:.2f}%")
            if best_acc > 0:
                self.logger.log_workflow(f"最佳查询集准确率: {best_acc:.2f}%")
        
        return max(query_acc, best_acc)
    
    def baseline_test(self, support_images, lora_images, query_images, num_epochs=None, callback=None, batch_size=None, use_minibatch=None, sample_type=None):
        """
        基准测试：不微调，直接使用预训练模型测试
        """
        # 数据准备
        support = support_images.reshape(-1, 3, 224, 224).cuda()
        query = query_images.reshape(-1, 3, 224, 224).cuda()
        
        # 准备标签
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        
        self.backbone.cuda()
        self.backbone.eval()
        
        self.logger.log_workflow("\n===== 基准测试（无微调）=====")
        
        # 使用支持集训练分类器头
        with torch.no_grad():
            support_features = self.backbone(support)
        
        # 使用支持集特征训练线性分类器
        self.fc = fc_block(support_features.size(1), self.n_way).cuda()
        optimizer = torch.optim.SGD(self.fc.parameters(), lr=0.01)
        
        # 训练分类器头部
        for epoch in range(100):  # 仅训练分类头
            optimizer.zero_grad()
            scores = self.fc(support_features)
            loss = self.loss_fn(scores, y_support)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.log_workflow(f"训练分类头 - Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # 测试
        with torch.no_grad():
            query_features = self.backbone(query)
            query_scores = self.fc(query_features)
            
            y_query_cal = np.repeat(range(self.n_way), self.n_query)
            topk_scores, topk_labels = query_scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query_cal)
            correct_this, count_this = float(top1_correct), len(y_query_cal)
            query_acc = float(correct_this) / float(count_this) * 100
            
            self.logger.log_workflow(f"基准测试准确率: {query_acc:.2f}%")
        
        return query_acc

    def support_only_fine_tune(self, support_images, lora_images, query_images, num_epochs=100, callback=None, batch_size=5, use_minibatch=True, sample_type=2):
        """
        只使用支持集图像进行微调，同时微调backbone和分类器头
        """
        # 数据准备
        support = support_images.reshape(-1, 3, 224, 224).cuda()
        query = query_images.reshape(-1, 3, 224, 224).cuda()
        np.random.seed(10)
        
        # 准备标签
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        
        # 初始化优化器 - 这里同时优化backbone和分类器
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
        
        self.backbone.cuda()
        # 确保backbone处于训练模式
        self.backbone.train()
        
        # 记录最佳模型
        best_acc = 0.0
        best_state = None
        
        self.logger.log_workflow("\n===== 仅使用支持集图像进行全模型微调 =====")
        for epoch in range(num_epochs):
            # 在每个epoch随机打乱支持集
            rand_idx = torch.randperm(support.size(0))
            shuffled_support = support[rand_idx]
            shuffled_y_support = y_support[rand_idx]
            
            if use_minibatch:
                # Mini-batch训练
                total_loss = 0.0
                for j in range(0, support.size(0), batch_size):
                    end_idx = min(j + batch_size, support.size(0))
                    batch_support = shuffled_support[j:end_idx]
                    batch_y_support = shuffled_y_support[j:end_idx]
                    
                    optimizer.zero_grad()
                    features = self.backbone(batch_support)
                    scores = self.fc(features)
                    loss = self.loss_fn(scores, batch_y_support)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / ((support.size(0) + batch_size - 1) // batch_size)
            else:
                # 完整批次训练
                optimizer.zero_grad()
                features = self.backbone(support)
                scores = self.fc(features)
                loss = self.loss_fn(scores, y_support)
                loss.backward()
                optimizer.step()
                avg_loss = loss.item()
            
            if epoch % 10 == 0:
                self.logger.log_workflow(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
                # 每30个epoch评估一次
                if epoch > 0 and epoch % 30 == 0:
                    self.backbone.eval()
                    with torch.no_grad():
                        # 在查询集上评估
                        query_features = self.backbone(query)
                        query_scores = self.fc(query_features)
                        preds = torch.argmax(query_scores, dim=1)
                        acc = (preds == y_query).float().mean().item() * 100
                        self.logger.log_workflow(f"  查询集准确率: {acc:.2f}%")
                        
                        # 保存最佳模型
                        if acc > best_acc:
                            best_acc = acc
                            best_state = {
                                'backbone': self.backbone.state_dict(),
                                'fc': self.fc.state_dict()
                            }
                    
                    self.backbone.train()
        
        # 恢复最佳模型
        if best_state is not None:
            self.backbone.load_state_dict(best_state['backbone'])
            self.fc.load_state_dict(best_state['fc'])
        
        # 最终评估
        self.backbone.eval()
        with torch.no_grad():
            query_features = self.backbone(query)
            query_scores = self.fc(query_features)
            
            y_query_cal = np.repeat(range(self.n_way), self.n_query)
            topk_scores, topk_labels = query_scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query_cal)
            correct_this, count_this = float(top1_correct), len(y_query_cal)
            query_acc = float(correct_this) / float(count_this) * 100
            
            self.logger.log_workflow(f"最终查询集准确率: {query_acc:.2f}%")
            if best_acc > 0:
                self.logger.log_workflow(f"最佳查询集准确率: {best_acc:.2f}%")
        
        return max(query_acc, best_acc)
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
