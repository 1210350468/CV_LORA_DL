"""
任务管理器
负责执行和管理测试任务
"""

import torch
import numpy as np
from utils.image_utils import save_images
from lora.lora_trainer import LoraTrainer

class TaskManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.lora_trainer = LoraTrainer(config, logger)
        
        self.status = {
            'current_task': 0,
            'total_tasks': config.task_num,
            'current_epoch': 0,
            'total_epochs': 100,
            'task_loss': 0.0,
            'task_acc': 0.0,
            'state': 'idle'
        }
        
    def execute_tasks(self, model, support_data, query_data):
        """执行所有测试任务"""
        accuracies = []
        
        for task_id in range(self.config.task_num):
            self.status['current_task'] = task_id + 1
            self.logger.log_workflow(f"Starting task {task_id + 1}/{self.config.task_num}")
            
            try:
                # 执行单个任务
                acc = self._execute_single_task(model, support_data, query_data, task_id)
                accuracies.append(acc)
                
                self.logger.log_workflow(f"Task {task_id + 1} completed with accuracy: {acc*100:.2f}%")
                
            except Exception as e:
                self.logger.log_error(f"Error in task {task_id + 1}: {str(e)}")
                continue
                
        # 计算统计结果
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        return mean_acc, std_acc
        
    def _execute_single_task(self, model, support_data, query_data, task_id):
        """执行单个任务"""
        # 保存支持集图像
        save_images(support_data, self.config.output_support_dir, task_id, self.logger)
        
        # 使用Lora训练
        self.lora_trainer.train(support_data)
        
        # 微调模型
        acc = model.fine_tune(support_data, query_data)
        
        return acc 