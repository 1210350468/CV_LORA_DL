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
from dataloader import EuroSAT_few_shot
import argparse
# 导入测试管理器
from TestManager import TestManager, delete_dir, split_support_query
from dataloader.FSLDataLoader import FSLTaskLoader, get_combined_support

def run_test(n_way=5, n_support=5, n_query=15, task_num=600, dataset="EuroSAT", model_name="ResNet18", mode="normal"):
    """
    主测试函数入口点
    使用FSLTaskLoader加载任务图像，然后进行微调测试
    
    参数:
    - n_way: 类别数量
    - n_support: 每类支持集样本数
    - n_query: 每类查询集样本数
    - task_num: 要测试的任务数量
    - dataset: 数据集名称
    - model_name: 模型名称
    - mode: 测试模式，可选'normal'(正常微调)、'lora_only'(仅LoRA微调)、'baseline'(不微调)、'support_only'(仅支持集微调)
    """
    np.random.seed(10)
    test_manager = TestManager()
    test_manager.config.update(
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        task_num=task_num,
        dataset=dataset,
        model_name=model_name
    )
    
    start_time = time.time()
    
    # 根据模式选择不同的测试方法
    if mode == "normal":
    mean_acc, std_acc = test_manager.run_finetune_test()
    elif mode == "lora_only":
        mean_acc, std_acc = test_manager.run_lora_only_test()
    elif mode == "baseline":
        mean_acc, std_acc = test_manager.run_baseline_test()
    elif mode == "support_only":
        mean_acc, std_acc = test_manager.run_support_only_test()
    else:
        raise ValueError(f"不支持的测试模式: {mode}，请选择'normal'、'lora_only'、'baseline'或'support_only'")
    
    end_time = time.time()
    
    acc_rate = mean_acc 
    std_rate = std_acc 
    
    # 打印测试结果和用时
    time_taken = datetime.timedelta(seconds=int(end_time - start_time))
    result_str = f"准确率: {acc_rate:.2f} % ± {1.96*std_rate/np.sqrt(task_num):.2f} %"
    time_str = f"测试用时: {time_taken}"
    mode_str = f"测试模式: {mode}"
    
    print(f"\n{'-'*50}")
    print(f"{mode_str}")
    print(f"{result_str}")
    print(f"{time_str}")
    print(f"{'-'*50}")
    
    return mean_acc, std_acc

if __name__=="__main__":
    # 读取命令行参数，如果有
    parser = argparse.ArgumentParser(description='运行微调测试')
    parser.add_argument('--n_way', type=int, default=5, help='类别数量')
    parser.add_argument('--n_support', type=int, default=5, help='每类支持集样本数')
    parser.add_argument('--n_query', type=int, default=15, help='每类查询集样本数')
    parser.add_argument('--task_num', type=int, default=100, help='测试任务数量')
    parser.add_argument('--dataset', type=str, default="EuroSAT", help='数据集名称')
    parser.add_argument('--model_name', type=str, default="ResNet18", help='模型名称')
    parser.add_argument('--mode', type=str, default="normal", choices=["normal", "lora_only", "baseline", "support_only"], help='测试模式')
    
    args = parser.parse_args()
    
    np.random.seed(10)
    mean_acc, std_acc = run_test(
        n_way=args.n_way, 
        n_support=args.n_support, 
        n_query=args.n_query, 
        task_num=args.task_num,
        dataset=args.dataset,
        model_name=args.model_name,
        mode=args.mode
    )
