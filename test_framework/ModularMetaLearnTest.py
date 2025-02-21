"""
模块化的元学习测试主文件
整合所有模块提供统一的测试接口
"""

import os
import sys
from core.task_manager import TaskManager
from core.model_manager import ModelManager
from core.data_manager import DataManager
from utils.log_utils import LogManager
from config.test_config import TestConfig

class ModularMetaLearnTest:
    def __init__(self):
        self.config = TestConfig()
        self.logger = LogManager()
        
        # 初始化各个管理器
        self.task_manager = TaskManager(self.config, self.logger)
        self.model_manager = ModelManager(self.config, self.logger)
        self.data_manager = DataManager(self.config, self.logger)
    def run_test(self):
        """运行测试的主函数"""
        try:
            # 准备数据
            support_data, query_data = self.data_manager.prepare_data()
            
            # 准备模型
            model = self.model_manager.prepare_model()
            
            # 执行任务
            results = self.task_manager.execute_tasks(model, support_data, query_data)
            
            return results
            
        except Exception as e:
            self.logger.log_error(f"Error in run_test: {str(e)}")
            raise

if __name__ == "__main__":
    test = ModularMetaLearnTest()
    mean_acc, std_acc = test.run_test()
    print(f"Final Results - Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%") 