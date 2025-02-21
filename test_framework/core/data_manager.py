"""
数据管理器
负责数据的加载、预处理和管理
"""

import torch
from torch.utils.data import DataLoader
from utils.data_utils import get_meta_dataloader

class DataManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def prepare_data(self):
        """准备数据"""
        self.logger.log_workflow("Preparing data...")
        
        try:
            # 获取数据加载器
            dataloader = get_meta_dataloader(
                n_way=self.config.n_way,
                n_support=self.config.n_support,
                n_query=self.config.n_query
            )
            
            # 获取一批数据
            support_data, query_data = next(iter(dataloader))
            
            self.logger.log_workflow(
                f"Data prepared: {self.config.n_way}-way "
                f"{self.config.n_support}-shot with "
                f"{self.config.n_query} query samples"
            )
            
            return support_data, query_data
            
        except Exception as e:
            self.logger.log_error(f"Error preparing data: {str(e)}")
            raise 