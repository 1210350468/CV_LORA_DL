"""
模型管理器
负责模型的加载、初始化和管理
"""

import torch
import torch.nn as nn
from models.backbone import get_backbone
from models.meta_model import get_meta_model

class ModelManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def prepare_model(self):
        """准备模型"""
        self.logger.log_workflow(f"Preparing {self.config.model_name} model...")
        
        try:
            # 获取backbone
            backbone = get_backbone(self.config.model_name)
            
            # 获取元学习模型
            model = get_meta_model(
                backbone=backbone,
                n_way=self.config.n_way,
                n_support=self.config.n_support
            )
            
            # 移动到GPU
            model = model.cuda()
            
            self.logger.log_workflow("Model prepared successfully")
            
            return model
            
        except Exception as e:
            self.logger.log_error(f"Error preparing model: {str(e)}")
            raise 