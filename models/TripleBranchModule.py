import torch
import torch.nn as nn
import torch.distributions as dist

class TripleBranchModule(nn.Module):
    def __init__(self, feature_dim, loss_fn, alpha=0.4):
        """
        三分支特征处理模块
        :param feature_dim: 输入特征维度
        :param loss_fn: 损失函数 (如 nn.CrossEntropyLoss)
        :param alpha: Beta 分布的超参数 (建议 0.2~0.4)
        """
        super(TripleBranchModule, self).__init__()
        self.alpha = alpha
        self.feature_dim = feature_dim
        
        # 保持原始特征的分支 - 使用残差连接
        self.branch_orig = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # LoRA 分支 - 使用残差连接
        self.branch_lora = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Mixup 分支 - 原本结构简单，这里添加残差连接
        self.branch_mixup = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 初始化接近恒等映射的权重（如果需要可以进一步调整）
        self._init_weights()

        self.loss_fn = loss_fn

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat_orig, feat_lora, epoch, logger):
        batch_size = feat_orig.shape[0]
        
        # 原始分支输出，添加残差连接
        out_orig = self.branch_orig(feat_orig) + feat_orig
        
        # LoRA 分支输出，添加残差连接
        out_lora = self.branch_lora(feat_lora) + feat_lora
        
        # Mixup 分支：使用 Beta 分布生成 mixup 权重
        beta_dist = dist.Beta(self.alpha, self.alpha)
        mixup_lambda = beta_dist.sample((batch_size,)).to(feat_orig.device).unsqueeze(1)
        feat_mixup = mixup_lambda * feat_orig + (1 - mixup_lambda) * feat_lora
        
        # 现在为 mixup 分支也添加残差连接
        out_mixup = self.branch_mixup(feat_mixup) + feat_mixup
        
        # if epoch == 0 or epoch % 10 == 0:
        #     with torch.no_grad():
        #         logger.log_workflow(f"""
        #         Feature statistics:
        #         - Original features: mean={feat_orig.mean().item():.4f}, std={feat_orig.std().item():.4f}
        #         - Branch outputs:
        #             orig: mean={out_orig.mean().item():.4f}, std={out_orig.std().item():.4f}
        #             lora: mean={out_lora.mean().item():.4f}, std={out_lora.std().item():.4f}
        #             mixup: mean={out_mixup.mean().item():.4f}, std={out_mixup.std().item():.4f}
        #         - Mixup lambda: {mixup_lambda.mean().item():.4f}
        #         """)
        
        return out_orig, out_lora, out_mixup
