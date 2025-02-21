#元测试，对特征提取器微调几个epoch后进行测试
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import os
import torchvision.models as models
from tqdm import tqdm
# 添加模块搜索路径
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
#添加日志库
import logging
import shutil
import datetime
import traceback
import time
import subprocess
from subprocess import PIPE, STDOUT
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Pretrain_ResNet_Adapt import get_test_model,load_pretrained_weights
from models.Pretrain_VIT import get_vit_model
from dataloader.Metastage_Dataloader import get_meta_dataloader
from utils.data_tagging import batch_annotate_and_save as tag_and_save
from utils.trainlora import trainlora as Lora_Train
from dataloader.Lora_Dataloader import Load_Lora_image
from config import TestConfig
from utils.log_manager import LogManager

class TestManager:
    def __init__(self):
        self.status = {
            'current_task': 0,
            'total_tasks': 0,
            'current_epoch': 0,
            'total_epochs': 100,
            'task_loss': 0.0,
            'task_acc': 0.0,
            'state': 'idle',
            'log_messages': []
        }
        
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 初始化配置
        self.config = TestConfig()
        
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
        
        # 确保其他日志文件存在
        for log_file in ['workflow.log', 'task_output.log']:
            log_path = os.path.join('logs', log_file)
            if not os.path.exists(log_path):
                open(log_path, 'a').close()
        
        # 初始化日志管理器
        self.logger = LogManager()
        
        # 确保输出目录存在
        os.makedirs(self.config.output_support_dir, exist_ok=True)
        os.makedirs(self.config.output_Lora_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def log_workflow(self, message):
        """记录工作流程日志"""
        self.logger.log_workflow(message)
        self._update_status_log(message)

    def _update_status_log(self, message):
        """更新状态日志（用于前端显示）"""
        if 'log_messages' not in self.status:
            self.status['log_messages'] = []
        self.status['log_messages'].append(message)

    def update_status(self, **kwargs):
        """更新状态"""
        self.status.update(kwargs)
        if 'state' in kwargs:
            self.logger.log_workflow(f"Test state changed to: {kwargs['state']}")
            self._update_status_log(f"State: {kwargs['state']}")

    def lora_train(self):
        """Lora训练方法"""
        self.logger.log_workflow("Starting Lora training...")
        original_dir = os.getcwd()
        
        try:
            lora_scripts_dir = "/root/autodl-tmp/all_starnight/lora-scripts"
            script_path = os.path.join(lora_scripts_dir, "train_by_toml.sh")
            
            self.logger.log_workflow(f"Using script: {script_path}")
            os.chdir(lora_scripts_dir)
            
            # 使用execute_command执行训练脚本
            cmd = f"bash {script_path}"
            result = self.execute_command(cmd, working_dir=lora_scripts_dir)
            
            if result != 0:
                error_msg = f"Lora training failed with return code {result}"
                self.logger.log_workflow_error(error_msg)
                raise Exception(error_msg)
            
            self.logger.log_workflow("Lora training completed successfully")
            
        except Exception as e:
            error_msg = f"Lora training failed: {str(e)}"
            self.logger.log_workflow_error(error_msg)
            raise
        finally:
            os.chdir(original_dir)

    def execute_command(self, cmd, working_dir=None):
        """执行命令并显示输出"""
        original_dir = os.getcwd()
        try:
            if working_dir:
                os.chdir(working_dir)
            
            # 直接执行命令，不捕获输出
            process = subprocess.Popen(
                cmd,
                shell=True,
                # 不设置stdout和stderr，让输出直接显示在终端
            )
            
            # 等待命令执行完成
            process.wait()
            return process.returncode
            
        finally:
            if working_dir:
                os.chdir(original_dir)

    def run_test(self, n_way=None, n_support=None, n_query=None, task_num=None, dataset=None):
        try:
            # 开始前清理所有目录
            # self.clean_directories()
            
            # 更新配置
            if any(x is not None for x in [n_way, n_support, n_query, task_num, dataset]):
                self.config.update(
                    n_way=n_way or self.config.n_way,
                    n_support=n_support or self.config.n_support,
                    n_query=n_query or self.config.n_query,
                    task_num=task_num or self.config.task_num,
                    dataset=dataset or self.config.dataset
                )
            
            self.logger.log_workflow(
                f"Starting test with {self.config.n_way}-way-{self.config.n_support}-shot "
                f"on {self.config.dataset} dataset"
            )
            
            # 构建数据集路径
            json_path = os.path.join(self.project_root, 'utils', f'{self.config.dataset}.json')
            
            # 添加数据加载日志
            self.logger.log_workflow(f"Loading data from {self.config.dataset} dataset...")
            
            test_loader = get_meta_dataloader(
                num_classes=self.config.n_way,
                num_support=self.config.n_support,
                num_query=self.config.n_query,
                task_num=self.config.task_num, 
                json_file=json_path
            )
            # 创建必要的目录
            output_support_dir = self.config.output_support_dir
            output_Lora_dir = self.config.output_Lora_dir
            output_images_dir = self.config.output_dir
            self.logger.log_workflow(f"Created directories: {output_support_dir}, {output_Lora_dir}")
            
            acc_list = []
            
            for i, (images, labels) in enumerate(test_loader):
                try:
                    # 每个task开始前，确保目录存在并且是空的
                    self.setup_task_directories()
                    
                    self.logger.log_workflow(f"Processing task {i+1}/{self.config.task_num} ({(i+1)/self.config.task_num*100:.1f}%)")
                    self.update_status(
                        current_task=i+1,
                        state='processing_task'
                    )
                    
                    # 选择模型
                    self.logger.log_workflow("Initializing model...")
                    model = get_test_model(
                        model_name='resnet18', 
                        num_classes=self.config.n_way, 
                        freeze_backbone=False,
                        logger=self.logger,
                        config=self.config  # 传入配置
                    )
                    model = model.cuda()
                    self.logger.log_workflow("Model moved to GPU")
                    
                    # 加载预训练模型
                    pretrained_path = os.path.join(self.project_root, 'output/pretrain/ResNet18/model_im1k.pth')
                    self.logger.log_workflow(f"Loading pretrained weights from {pretrained_path}")
                    load_pretrained_weights(model, pretrained_path)
                    self.logger.log_workflow("Pretrained weights loaded")
                    
                    # 准备数据
                    self.logger.log_workflow("Preparing support and query sets...")
                    support, query, support_labels, query_labels = split_support_query(images, labels, self.config.n_support, self.config.n_query)
                    self.logger.log_workflow(f"Support set shape: {support.shape}, Query set shape: {query.shape}")
                    
                    # 保存support图片
                    self.logger.log_workflow("Saving support images...")
                    save_images(support, support_labels, output_support_dir, self.config.label_set, i, self)
                    self.logger.log_workflow("Support images saved")
                    
                    # 标注图片
                    self.logger.log_workflow("Starting image annotation...")
                    tag_and_save(output_support_dir, output_support_dir)
                    self.logger.log_workflow("Image annotation completed")
                    
                    # 启动Lora训练
                    # self.logger.log_workflow("Starting Lora training...")
                    # self.lora_train()
                    # self.logger.log_workflow("Lora training completed")
                    
                    # 生成新图片
                    self.logger.log_workflow("Generating new images with Lora...")
                    cmd = "python ../utils/create_image.py"
                    result = self.execute_command(cmd)
                    if result != 0:
                        raise Exception(f"Image generation failed with return code {result}")
                    self.logger.log_workflow("Image generation completed")
                    
                    # 加载Lora生成的图片
                    self.logger.log_workflow("Loading Lora generated images...")
                    Lora_images, Lora_labels = Load_Lora_image(label_set=self.config.label_set, original_labels_tensor=support_labels)
                    self.logger.log_workflow("Lora images loaded")
                    
                    # 合并原始和Lora图片
                    self.logger.log_workflow("Combining original and Lora images...")
                    finetune_images = torch.cat([support, Lora_images], dim=1)
                    self.logger.log_workflow(f"Combined images shape: {finetune_images.shape}")
                    
                    def epoch_callback(epoch, total_epochs, loss, acc):
                        """每个epoch的回调函数"""
                        self.status.update({
                            'current_epoch': epoch,
                            'task_loss': loss,
                            'task_acc': acc
                        })
                        # 记录到task日志
                        self.logger.log_task(i+1, epoch, loss, acc)
                        # 更新前端显示
                        self._update_status_log(
                            f"Task {i+1}, Epoch {epoch}/{total_epochs}: "
                            f"loss={loss:.4f}, accuracy={acc*100:.2f}%"
                        )
                    
                    # 开始微调
                    self.logger.log_workflow(f"Starting fine-tuning for task {i+1}")
                    acc = model.fine_tune(
                        finetune_images,
                        query,
                        num_epochs=100,
                        callback=epoch_callback
                    )
                    
                    # 记录任务结果
                    self.logger.log_task_summary(i+1, acc)
                    acc_list.append(acc)
                    # 修改结果保存的调用
                    self.logger.save_task_result(
                        task_id=i+1,
                        accuracy=acc,
                        details={
                            'epoch_count': 100,
                            'dataset': self.config.dataset,
                            'n_way': self.config.n_way,
                            'n_support': self.config.n_support,
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'support_images_count': finetune_images.size(1),
                            'query_images_count': query.size(1)
                        }
                    )
                    
                    # 任务完成后清理
                    self.clean_directories()
                    
                except Exception as e:
                    self.logger.log_workflow(f"Error in task {i+1}: {str(e)}")
                    self.clean_directories()
                    raise
            
            # 计算最终结果
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            
            final_result = (
                f"Test completed: {self.config.n_way}-way-{self.config.n_support}-shot, "
                f"dataset: {self.config.dataset}\n"
                f"Final accuracy: {mean_acc*100:.2f}% ± "
                f"{1.96*std_acc*100/np.sqrt(self.config.task_num):.2f}%"
            )
            
            self.logger.log_workflow(final_result)
            
            self.update_status(
                state='completed',
                final_accuracy=float(mean_acc),
                std_accuracy=float(std_acc)
            )
            
            return mean_acc, std_acc
            
        except Exception as e:
            self.logger.log_workflow(f"Error during test: {str(e)}")
            self.clean_directories()
            raise

    def clean_directories(self):
        """清理所有临时目录的内容"""
        directories = [
            self.config.output_support_dir,  # support_images/5_zkz
            self.config.output_Lora_dir,     # support_images/Lora_images
        ]
        
        for directory in directories:
            if os.path.exists(directory):
                try:
                    # 只清理目录中的内容，保留目录结构
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            self.logger.log_workflow(f"Cleaned content: {file_path}")
                        except Exception as e:
                            self.logger.log_workflow(f"Error cleaning {file_path}: {e}")
                except Exception as e:
                    self.logger.log_workflow(f"Error accessing directory {directory}: {e}")

    def setup_task_directories(self):
        """确保任务所需的目录存在且为空"""
        # 首先确保父目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        directories = [
            self.config.output_support_dir,  # support_images/5_zkz
            self.config.output_Lora_dir,     # support_images/Lora_images
        ]
        
        for directory in directories:
            # 如果目录存在，清空其中的内容
            if os.path.exists(directory):
                try:
                    # 只删除目录中的内容，不删除目录本身
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            self.logger.log_workflow(f"Cleaned content: {file_path}")
                        except Exception as e:
                            self.logger.log_workflow(f"Error cleaning {file_path}: {e}")
                except Exception as e:
                    self.logger.log_workflow(f"Error accessing directory {directory}: {e}")
            
            # 确保目录存在
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.log_workflow(f"Ensured directory exists: {directory}")
            except Exception as e:
                self.logger.log_workflow(f"Error creating directory {directory}: {e}")

def split_support_query(data, labels, n_support=5, n_query=15):
    support = data[:, :n_support, :, :, :]
    query = data[:, n_support:, :, :, :]
    support_labels = labels[:, :n_support]
    query_labels = labels[:, n_support:]
    return support, query, support_labels, query_labels

def delete_folders(path):
    """删除指定路径下的所有内容"""
    try:
        if os.path.exists(path):
            # 先删除文件
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

            # 再删除文件夹
            for root, dirs, files in os.walk(path, topdown=False):
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"Error deleting directory {dir_path}: {e}")
    except Exception as e:
        print(f"Error in delete_folders for path {path}: {e}")

# 添加保存图像的函数
def save_images(data, label, output_dir, label_set, task_id, manager):
    """保存图像的函数
    Args:
        data: 图像数据
        label: 标签数据
        output_dir: 输出目录
        label_set: 标签集
        task_id: 任务ID
        manager: TestManager实例
    """
    manager.logger.log_workflow(f"Saving images to {output_dir}")
    manager.logger.log_workflow(f"Data shape: {data.shape}")
    manager.logger.log_workflow(f"Labels shape: {label.shape}")
    
    for j in range(data.size(0)):
        for k in range(data.size(1)):
            try:
                # 撤销归一化
                unnormalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
                img_array = data[j][k]

                for t, m, s in zip(img_array, unnormalize['mean'], unnormalize['std']):
                    t.mul_(s).add_(m)  # 反归一化

                # 将张量从 (C, H, W) 转换为 (H, W, C)
                img_array = img_array.permute(1, 2, 0).numpy()

                # 确保图像数据在 [0, 255] 范围内并转换为 uint8
                img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

                # 转换为 PIL 图像
                img = Image.fromarray(img_array)
                img = img.resize((256, 256), Image.LANCZOS)
                
                # 生成文件名
                label_name = label_set[label[j][k].item()]
                file_name = f"{output_dir}/{label_name}_{task_id}{j}{k}.png"

                # 保存图像
                img.save(file_name)
                print(f"Saved {file_name}")
                manager.logger.log_workflow(f"Saved image {j},{k} with label {label_name}")
            except Exception as e:
                manager.logger.log_workflow(f"Error saving image {j},{k}: {str(e)}")

if __name__=="__main__":
    # 配置日志记录
    logging.basicConfig(filename='test_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    test_manager = TestManager()
    mean_acc, std_acc = test_manager.run_test()
    acc_rate = mean_acc * 100
    std_rate = std_acc * 100

    print(f"accuracy: {acc_rate:.2f} % ± {1.96*std_rate/np.sqrt(test_manager.config.task_num):.2f} %")
    logging.info(f"Test {test_manager.config.n_way}-way-{test_manager.config.n_support}-shot,dataset : {test_manager.config.dataset},task_num = {test_manager.config.task_num},accuracy: {acc_rate:.2f} % ± {1.96*std_rate/np.sqrt(test_manager.config.task_num):.2f} %")
