"""
日志工具
提供日志记录和管理功能
"""

import logging
import os
import json
from datetime import datetime

class LogManager:
    def __init__(self):
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 配置日志记录器
        self.setup_loggers()
        
        # 初始化结果记录
        self.init_results_file()
        
    def setup_loggers(self):
        """设置日志记录器"""
        # 工作流程日志
        self.workflow_logger = logging.getLogger('workflow')
        self.setup_logger(self.workflow_logger, 'logs/workflow.log')
        
        # 任务日志
        self.task_logger = logging.getLogger('task')
        self.setup_logger(self.task_logger, 'logs/task_output.log')
        
    def setup_logger(self, logger, filename):
        """配置单个日志记录器"""
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    def init_results_file(self):
        """初始化结果文件"""
        results_path = 'logs/results.json'
        if not os.path.exists(results_path):
            initial_results = {
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_tasks': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'tasks': []
            }
            with open(results_path, 'w') as f:
                json.dump(initial_results, f, indent=4)
                
    def log_workflow(self, message):
        """记录工作流程日志"""
        self.workflow_logger.info(message)
        
    def log_task(self, message):
        """记录任务日志"""
        self.task_logger.info(message)
        
    def log_error(self, message):
        """记录错误日志"""
        self.workflow_logger.error(message)
        self.task_logger.error(message) 