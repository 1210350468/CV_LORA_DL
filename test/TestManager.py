import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import torchvision.models as models
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import logging
import shutil
import datetime
import traceback
import time
import subprocess
from subprocess import PIPE, STDOUT
import json

# 添加模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader.Metastage_Dataloader import get_meta_dataloader
from models.Pretrain_ResNet_Adapt import get_test_model, load_pretrained_weights
from models.Pretrain_VIT import get_vit_model
from utils.data_tagging import batch_annotate_and_save as tag_and_save
from utils.trainlora import trainlora as Lora_Train
from dataloader.Lora_Dataloader import Load_Lora_image
from config import TestConfig
from dataloader.FSLDataLoader import FSLTaskLoader, get_combined_support

class TestManager:
    def __init__(self):
        # self.status = {
        #     'current_task': 0,
        #     'total_tasks': 0,
        #     'current_epoch': 0,
        #     'total_epochs': 100,
        #     'task_loss': 0.0,
        #     'task_acc': 0.0,
        #     'state': 'idle'
        # }
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 初始化配置
        self.config = TestConfig()
        self.json_path = os.path.join(self.project_root, 'utils', f'{self.config.dataset}.json')
        # 确保 results.json 存在且有初始结构
        results_path = os.path.join('logs', 'results.json')
        if not os.path.exists(results_path):
            initial_results = {
                'last_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_tasks': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 1.0,
                'tasks': []
            }
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(initial_results, f, indent=4)
        
        # 初始化日志管理器（仅保留JSON日志记录功能）
        self.logger = JsonLogger()
        
        # 确保输出目录存在
        os.makedirs(self.config.output_support_dir, exist_ok=True)
        os.makedirs(self.config.output_Lora_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def log_message(self, message):
        """简单的日志记录函数"""
        print(message)
        
    def make_task_images(self):
        """从支持集生成Lora图像,并且保存n个task的support和query，目录是task_images_dir/task_id/class_id/[support,query,lora]/image_name.png，
        每次lora训练生图工作流传入一个task的support和support_labels，需要使用到dataloader"""
        # 确保主目录存在
        os.makedirs(self.config.task_images_dir, exist_ok=True)
        
        test_loader = get_meta_dataloader(
            num_classes=self.config.n_way,
            num_support=self.config.n_support,
            num_query=self.config.n_query,
            task_num=self.config.task_num, 
            json_file=self.json_path
        )
        for i, (images, labels) in enumerate(test_loader):
            # 准备数据
            support, query, support_labels, query_labels = split_support_query(
                images, labels, self.config.n_support, self.config.n_query
            )
            lora_images, lora_labels = self.run_lora_workflow(support, support_labels)
            
            # 创建当前任务目录
            current_task_dir = os.path.join(self.config.task_images_dir, f'task_{i}')
            os.makedirs(current_task_dir, exist_ok=True)
            
            # 保存不同类型的图像
            self.log_message(f"保存任务 {i} 的支持集图像")
            save_images_with_custom_dir(support, support_labels, self.config.task_images_dir, self.config.label_set, f'task_{i}', 'support', self)
            
            self.log_message(f"保存任务 {i} 的查询集图像")
            save_images_with_custom_dir(query, query_labels, self.config.task_images_dir, self.config.label_set, f'task_{i}', 'query', self)
            
            self.log_message(f"保存任务 {i} 的Lora图像")
            save_images_with_custom_dir(lora_images, lora_labels, self.config.task_images_dir, self.config.label_set, f'task_{i}', 'lora', self)
            
    def run_finetune_test(self, n_way=None, n_support=None, n_query=None, task_num=None, dataset=None):
        """核心微调测试函数，与其他功能解耦"""
        try:
            # 更新配置
            if any(x is not None for x in [n_way, n_support, n_query, task_num, dataset]):
                self.config.update(
                    n_way=n_way or self.config.n_way,
                    n_support=n_support or self.config.n_support,
                    n_query=n_query or self.config.n_query,
                    task_num=task_num or self.config.task_num,
                    dataset=dataset or self.config.dataset
                )
            
            self.log_message(f"开始测试：{self.config.n_way}-way-{self.config.n_support}-shot，数据集：{self.config.dataset}")
            
            # 使用FSLTaskLoader加载数据
            self.log_message(f"加载 {self.config.dataset} 数据集，使用FSLTaskLoader")
            
            # 创建FSLTaskLoader
            task_loader = FSLTaskLoader(
                dataset_path="./task_images",  # 任务图像的路径
                n_way=self.config.n_way,
                n_support=self.config.n_support,
                n_query=self.config.n_query,
                include_lora=True  # 包含LoRA图像
            )
            
            # 任务数量
            total_tasks = len(task_loader)
            task_num = min(self.config.task_num, total_tasks)
            self.log_message(f"总共加载了 {total_tasks} 个任务，将测试 {task_num} 个任务")
            
            acc_list = []
            task_count = 0
            
            # 直接迭代任务加载器
            for support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info in task_loader:
                try:
                    # 如果达到指定任务数，退出循环
                    if task_count >= task_num:
                        break
                        
                    self.log_message(f"处理任务 {task_count+1}/{task_num} ({(task_count+1)/task_num*100:.1f}%)")
                    
                    # 初始化并加载模型
                    model = self._init_model()
                    
                    self.log_message(f"支持集形状: {support_images.shape}, 查询集形状: {query_images.shape}")
                    self.log_message(f"任务名称: {task_info['task_name']}")
                    
                    # 微调模型
                    acc = model.fine_tune(
                        support_images,
                        lora_images,
                        query_images,
                        num_epochs=100,
                    )
                    
                    # 记录结果
                    self.logger.save_task_result(
                        task_id=task_count+1,
                        accuracy=acc,
                        details={
                            'epoch_count': 100,
                            'dataset': self.config.dataset,
                            'n_way': self.config.n_way,
                            'n_support': self.config.n_support,
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query_images_count': query_images.size(1),
                            'task_name': task_info['task_name'],
                            'mode': 'normal'
                        }
                    )
                    
                    acc_list.append(acc)
                    task_count += 1
                    
                except Exception as e:
                    self.log_message(f"任务 {task_count+1} 出错: {str(e)}")
                    raise
            
            # 计算最终结果
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            
            final_result = (
                f"测试完成: {self.config.n_way}-way-{self.config.n_support}-shot, "
                f"数据集: {self.config.dataset}\n"
                f"最终准确率: {mean_acc:.4f} ± "
                f"{1.96*std_acc/np.sqrt(task_count):.4f}"
            )
            
            self.log_message(final_result)
            
            return mean_acc, std_acc
            
        except Exception as e:
            self.log_message(f"测试过程中出错: {str(e)}")
            raise

    def run_lora_only_test(self, n_way=None, n_support=None, n_query=None, task_num=None, dataset=None):
        """仅使用LoRA图像进行微调测试"""
        try:
            # 更新配置
            if any(x is not None for x in [n_way, n_support, n_query, task_num, dataset]):
                self.config.update(
                    n_way=n_way or self.config.n_way,
                    n_support=n_support or self.config.n_support,
                    n_query=n_query or self.config.n_query,
                    task_num=task_num or self.config.task_num,
                    dataset=dataset or self.config.dataset
                )
            
            self.log_message(f"开始仅LoRA微调测试：{self.config.n_way}-way-{self.config.n_support}-shot，数据集：{self.config.dataset}")
            
            # 使用FSLTaskLoader加载数据
            self.log_message(f"加载 {self.config.dataset} 数据集，使用FSLTaskLoader")
            
            # 创建FSLTaskLoader
            task_loader = FSLTaskLoader(
                dataset_path="./task_images",
                n_way=self.config.n_way,
                n_support=self.config.n_support,
                n_query=self.config.n_query,
                include_lora=True
            )
            
            # 任务数量
            total_tasks = len(task_loader)
            task_num = min(self.config.task_num, total_tasks)
            self.log_message(f"总共加载了 {total_tasks} 个任务，将测试 {task_num} 个任务")
            
            acc_list = []
            task_count = 0
            
            # 直接迭代任务加载器
            for support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info in task_loader:
                try:
                    # 如果达到指定任务数，退出循环
                    if task_count >= task_num:
                        break
                        
                    self.log_message(f"处理任务 {task_count+1}/{task_num} ({(task_count+1)/task_num*100:.1f}%)")
                    
                    # 初始化并加载模型
                    model = self._init_model()
                    
                    self.log_message(f"LoRA图像形状: {lora_images.shape}, 查询集形状: {query_images.shape}")
                    self.log_message(f"任务名称: {task_info['task_name']}")
                    
                    # 仅使用LoRA微调模型
                    acc = model.lora_only_fine_tune(
                        support_images,
                        lora_images,
                        query_images,
                        num_epochs=100,
                    )
                    
                    # 记录结果
                    self.logger.save_task_result(
                        task_id=task_count+1,
                        accuracy=acc,
                        details={
                            'epoch_count': 100,
                            'dataset': self.config.dataset,
                            'n_way': self.config.n_way,
                            'n_support': self.config.n_support,
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query_images_count': query_images.size(1),
                            'task_name': task_info['task_name'],
                            'mode': 'lora_only'
                        }
                    )
                    
                    acc_list.append(acc)
                    task_count += 1
                    
                except Exception as e:
                    self.log_message(f"任务 {task_count+1} 出错: {str(e)}")
                    raise
            
            # 计算最终结果
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            
            final_result = (
                f"仅LoRA测试完成: {self.config.n_way}-way-{self.config.n_support}-shot, "
                f"数据集: {self.config.dataset}\n"
                f"最终准确率: {mean_acc:.4f} ± "
                f"{1.96*std_acc/np.sqrt(task_count):.4f}"
            )
            
            self.log_message(final_result)
            
            return mean_acc, std_acc
            
        except Exception as e:
            self.log_message(f"测试过程中出错: {str(e)}")
            raise
    
    def run_baseline_test(self, n_way=None, n_support=None, n_query=None, task_num=None, dataset=None):
        """执行基准测试（不微调）"""
        try:
            # 更新配置
            if any(x is not None for x in [n_way, n_support, n_query, task_num, dataset]):
                self.config.update(
                    n_way=n_way or self.config.n_way,
                    n_support=n_support or self.config.n_support,
                    n_query=n_query or self.config.n_query,
                    task_num=task_num or self.config.task_num,
                    dataset=dataset or self.config.dataset
                )
            
            self.log_message(f"开始基准测试（无微调）：{self.config.n_way}-way-{self.config.n_support}-shot，数据集：{self.config.dataset}")
            
            # 使用FSLTaskLoader加载数据
            self.log_message(f"加载 {self.config.dataset} 数据集，使用FSLTaskLoader")
            
            # 创建FSLTaskLoader
            task_loader = FSLTaskLoader(
                dataset_path="./task_images",
                n_way=self.config.n_way,
                n_support=self.config.n_support,
                n_query=self.config.n_query,
                include_lora=True
            )
            
            # 任务数量
            total_tasks = len(task_loader)
            task_num = min(self.config.task_num, total_tasks)
            self.log_message(f"总共加载了 {total_tasks} 个任务，将测试 {task_num} 个任务")
            
            acc_list = []
            task_count = 0
            
            # 直接迭代任务加载器
            for support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info in task_loader:
                try:
                    # 如果达到指定任务数，退出循环
                    if task_count >= task_num:
                        break
                        
                    self.log_message(f"处理任务 {task_count+1}/{task_num} ({(task_count+1)/task_num*100:.1f}%)")
                    
                    # 初始化并加载模型
                    model = self._init_model()
                    
                    self.log_message(f"支持集形状: {support_images.shape}, 查询集形状: {query_images.shape}")
                    self.log_message(f"任务名称: {task_info['task_name']}")
                    
                    # 执行基准测试（不微调）
                    acc = model.baseline_test(
                        support_images,
                        lora_images,
                        query_images
                    )
                    
                    # 记录结果
                    self.logger.save_task_result(
                        task_id=task_count+1,
                        accuracy=acc,
                        details={
                            'dataset': self.config.dataset,
                            'n_way': self.config.n_way,
                            'n_support': self.config.n_support,
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query_images_count': query_images.size(1),
                            'task_name': task_info['task_name'],
                            'mode': 'baseline'
                        }
                    )
                    
                    acc_list.append(acc)
                    task_count += 1
                    
                except Exception as e:
                    self.log_message(f"任务 {task_count+1} 出错: {str(e)}")
                    raise
            
            # 计算最终结果
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            
            final_result = (
                f"基准测试完成: {self.config.n_way}-way-{self.config.n_support}-shot, "
                f"数据集: {self.config.dataset}\n"
                f"最终准确率: {mean_acc:.4f} ± "
                f"{1.96*std_acc/np.sqrt(task_count):.4f}"
            )
            
            self.log_message(final_result)
            
            return mean_acc, std_acc
            
        except Exception as e:
            self.log_message(f"测试过程中出错: {str(e)}")
            raise

    def run_support_only_test(self, n_way=None, n_support=None, n_query=None, task_num=None, dataset=None):
        """执行仅支持集微调测试（微调backbone和分类器头）"""
        try:
            # 更新配置
            if any(x is not None for x in [n_way, n_support, n_query, task_num, dataset]):
                self.config.update(
                    n_way=n_way or self.config.n_way,
                    n_support=n_support or self.config.n_support,
                    n_query=n_query or self.config.n_query,
                    task_num=task_num or self.config.task_num,
                    dataset=dataset or self.config.dataset
                )
            
            self.log_message(f"开始仅支持集微调测试：{self.config.n_way}-way-{self.config.n_support}-shot，数据集：{self.config.dataset}")
            
            # 使用FSLTaskLoader加载数据
            self.log_message(f"加载 {self.config.dataset} 数据集，使用FSLTaskLoader")
            
            # 创建FSLTaskLoader
            task_loader = FSLTaskLoader(
                dataset_path="./task_images",
                n_way=self.config.n_way,
                n_support=self.config.n_support,
                n_query=self.config.n_query,
                include_lora=True  # 虽然只用支持集，但为保持一致性仍加载LoRA
            )
            
            # 任务数量
            total_tasks = len(task_loader)
            task_num = min(self.config.task_num, total_tasks)
            self.log_message(f"总共加载了 {total_tasks} 个任务，将测试 {task_num} 个任务")
            
            acc_list = []
            task_count = 0
            
            # 直接迭代任务加载器
            for support_images, support_labels, lora_images, lora_labels, query_images, query_labels, task_info in task_loader:
                try:
                    # 如果达到指定任务数，退出循环
                    if task_count >= task_num:
                        break
                        
                    self.log_message(f"处理任务 {task_count+1}/{task_num} ({(task_count+1)/task_num*100:.1f}%)")
                    
                    # 初始化并加载模型
                    model = self._init_model()
                    
                    self.log_message(f"支持集形状: {support_images.shape}, 查询集形状: {query_images.shape}")
                    self.log_message(f"任务名称: {task_info['task_name']}")
                    
                    # 仅使用支持集微调模型
                    acc = model.support_only_fine_tune(
                        support_images,
                        lora_images,
                        query_images,
                        num_epochs=100,
                    )
                    
                    # 记录结果
                    self.logger.save_task_result(
                        task_id=task_count+1,
                        accuracy=acc,
                        details={
                            'epoch_count': 100,
                            'dataset': self.config.dataset,
                            'n_way': self.config.n_way,
                            'n_support': self.config.n_support,
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query_images_count': query_images.size(1),
                            'task_name': task_info['task_name'],
                            'mode': 'support_only'
                        }
                    )
                    
                    acc_list.append(acc)
                    task_count += 1
                    
                except Exception as e:
                    self.log_message(f"任务 {task_count+1} 出错: {str(e)}")
                    raise
            
            # 计算最终结果
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            
            final_result = (
                f"仅支持集微调测试完成: {self.config.n_way}-way-{self.config.n_support}-shot, "
                f"数据集: {self.config.dataset}\n"
                f"最终准确率: {mean_acc:.4f} ± "
                f"{1.96*std_acc/np.sqrt(task_count):.4f}"
            )
            
            self.log_message(final_result)
            
            return mean_acc, std_acc
            
        except Exception as e:
            self.log_message(f"测试过程中出错: {str(e)}")
            raise

    def _init_model(self):
        """初始化模型并加载预训练权重"""
        self.log_message("初始化模型...")
        model = get_test_model(
            model_name=self.config.model_name, 
            num_classes=self.config.n_way, 
            freeze_backbone=False,
            logger=self.logger,
            config=self.config
        )
        model = model.cuda()
        
        # 加载预训练模型
        pretrained_path = os.path.join(self.project_root, 'output/pretrain/ResNet18/model_im1k.pth')
        self.log_message(f"加载预训练权重: {pretrained_path}")
        load_pretrained_weights(model, pretrained_path, File_format='pth')
        
        return model

    def run_lora_workflow(self, support, support_labels):
        """运行Lora工作流：保存图像、标注、训练Lora、生成图像"""
        try:
            delete_dir(self.config.output_dir)
            # 保存support图片
            self.log_message("保存支持集图像...")
            save_images(support, support_labels, self.config.output_support_dir, self.config.label_set, 0, self)
            
            # 标注图片
            self.log_message("开始图像标注...")
            tag_and_save(self.config.output_support_dir, self.config.output_support_dir)
            
            # 启动Lora训练
            self.log_message("开始Lora训练...")
            self._run_lora_training()
            
            # 生成新图片
            self.log_message("使用Lora生成新图像...")
            cmd = "python ../utils/create_image.py"
            result = self._execute_command(cmd)
            if result != 0:
                raise Exception(f"图像生成失败，返回码 {result}")
            
            # 加载Lora生成的图片
            self.log_message("加载Lora生成的图像...")
            lora_images, lora_labels = Load_Lora_image(
                label_set=self.config.label_set, 
                original_labels_tensor=support_labels
            )
            #删除临时文件夹
            delete_dir(self.config.output_dir)
            return lora_images, lora_labels
            
        except Exception as e:
            self.log_message(f"Lora工作流出错: {str(e)}")
            raise

    def _run_lora_training(self):
        """运行Lora训练"""
        original_dir = os.getcwd()
        
        try:
            lora_scripts_dir = "/root/autodl-tmp/all_starnight/lora-scripts"
            script_path = os.path.join(lora_scripts_dir, "train_by_toml.sh")
            
            self.log_message(f"使用脚本: {script_path}")
            os.chdir(lora_scripts_dir)
            
            # 执行训练脚本
            cmd = f"bash {script_path}"
            result = self._execute_command(cmd, working_dir=lora_scripts_dir)
            
            if result != 0:
                raise Exception(f"Lora训练失败，返回码 {result}")
            
            self.log_message("Lora训练成功完成")
            
        except Exception as e:
            raise
        finally:
            os.chdir(original_dir)

    def _execute_command(self, cmd, working_dir=None):
        """执行命令并显示输出"""
        original_dir = os.getcwd()
        try:
            if working_dir:
                os.chdir(working_dir)
            
            process = subprocess.Popen(
                cmd,
                shell=True,
            )
            process.wait()
            return process.returncode
            
        finally:
            if working_dir:
                os.chdir(original_dir)

class JsonLogger:
    """简化的日志记录类，只保留JSON日志功能"""
    def __init__(self):
        self.results_path = os.path.join('logs', 'results.json')
        os.makedirs('logs', exist_ok=True)
        
        # 确保结果文件存在
        if not os.path.exists(self.results_path):
            initial_results = {
                'last_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_tasks': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 1.0,
                'tasks': []
            }
            with open(self.results_path, 'w', encoding='utf-8') as f:
                json.dump(initial_results, f, indent=4)
    
    def log_workflow(self, message):
        """记录工作流信息"""
        print(message)
        
    def log_task_summary(self, task_id, accuracy):
        """记录任务摘要"""
        print(f"任务 {task_id} 完成。 准确率: {accuracy:.2f}%")
    
    def save_task_result(self, task_id, accuracy, details=None):
        """保存任务结果到JSON文件"""
        try:
            # 读取现有结果
            with open(self.results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 更新结果
            task_result = {
                'task_id': task_id,
                'accuracy': float(accuracy),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if details:
                task_result.update(details)
            
            # 检查是否已存在相同ID的任务
            task_exists = False
            for i, task in enumerate(results['tasks']):
                if task.get('task_id') == task_id:
                    results['tasks'][i] = task_result
                    task_exists = True
                    break
            
            if not task_exists:
                results['tasks'].append(task_result)
                results['total_tasks'] = len(results['tasks'])
            
            # 重新计算统计信息
            accuracies = [task['accuracy'] for task in results['tasks']]
            if accuracies:
                results['mean_accuracy'] = float(np.mean(accuracies))
                results['std_accuracy'] = float(np.std(accuracies))
                results['best_accuracy'] = float(max(accuracies))
                results['worst_accuracy'] = float(min(accuracies))
            
            results['last_update'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 保存结果
            with open(self.results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
        except Exception as e:
            print(f"保存任务结果出错: {str(e)}")

# 辅助函数
def split_support_query(data, labels, n_support=5, n_query=15):
    """分割支持集和查询集"""
    support = data[:, :n_support, :, :, :]
    query = data[:, n_support:, :, :, :]
    support_labels = labels[:, :n_support]
    query_labels = labels[:, n_support:]
    return support, query, support_labels, query_labels

def delete_dir(dir_path):
    '''
    删除目录，如果目录存在
    '''
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def unnormalize_image(img):
    '''
    取消归一化
    '''
    # 确保输入是PyTorch张量
    if not torch.is_tensor(img):
        img = torch.tensor(img)
    
    # 确保数据在GPU上还是CPU上保持一致
    device = img.device
    
    # 定义均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    
    # 调整均值和标准差的维度以匹配图像张量
    if img.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    # 应用反归一化
    img = img * std + mean
    
    # 转换为numpy数组并调整通道顺序
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # 转换为PIL图像
    img = Image.fromarray(img)
    img = img.resize((256, 256), Image.LANCZOS)
    return img

def save_images_with_custom_dir(data, label, output_dir, label_set, task_id, type, manager):
    '''
    保存图片到output_dir/task_id/class_id/[support,query,lora]/image_name.png，
    每个图片张量都是[n_way,k_shot,3,224,224]，label是[n_way,k_shot]
    '''
    manager.log_message(f"保存图像到 {output_dir}")
    
    # 先取消归一化
    for i in range(data.size(0)):
        for j in range(data.size(1)):
            try:
                img = unnormalize_image(data[i][j].clone())
                
                # 安全地获取标签索引
                if torch.is_tensor(label[i][j]):
                    # 如果是张量，转换为标量
                    label_idx = label[i][j].item() if label[i][j].numel() == 1 else int(label[i][j][0])
                else:
                    # 已经是标量
                    label_idx = int(label[i][j])
                
                # 获取标签名称
                label_name = label_set[label_idx]
                
                # 创建目录路径
                # 确保task_id是字符串
                task_id_str = str(task_id)
                image_dir = os.path.join(output_dir, task_id_str, label_name, type)
                os.makedirs(image_dir, exist_ok=True)
                
                # 生成文件名
                file_name = os.path.join(image_dir, f"{i}_{j}.png")
                
                # 保存图像
                img.save(file_name)
                manager.log_message(f"已保存图像 {i},{j} 到 {file_name}")
            except Exception as e:
                manager.log_message(f"保存图像 {i},{j} 出错: {str(e)}")
                # 打印更详细的错误信息以便调试
                import traceback
                manager.log_message(traceback.format_exc())

def save_images(data, label, output_dir, label_set, task_id, manager):
    """保存图像函数"""
    manager.log_message(f"保存图像到 {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for j in range(data.size(0)):
        for k in range(data.size(1)):
            try:
                # 撤销归一化
                img = unnormalize_image(data[j][k].clone())
                
                # 安全地获取标签索引
                if torch.is_tensor(label[j][k]):
                    # 如果是张量，转换为标量
                    label_idx = label[j][k].item() if label[j][k].numel() == 1 else int(label[j][k][0])
                else:
                    # 已经是标量
                    label_idx = int(label[j][k])
                
                # 获取标签名称
                label_name = label_set[label_idx]
                
                # 生成文件名
                file_name = f"{output_dir}/{label_name}_{task_id}{j}{k}.png"

                # 保存图像
                img.save(file_name)
                manager.log_message(f"已保存图像 {j},{k}，标签为 {label_name}")
            except Exception as e:
                manager.log_message(f"保存图像 {j},{k} 出错: {str(e)}")
                # 打印更详细的错误信息以便调试
                manager.log_message(traceback.format_exc()) 