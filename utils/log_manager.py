import logging
import os
import json
import datetime
from logging.handlers import RotatingFileHandler
from collections import deque

class LogManager:
    def __init__(self):
        """初始化日志管理器"""
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 设置日志文件路径
        self.workflow_log = os.path.join('logs', 'workflow.log')
        self.task_log = os.path.join('logs', 'task_output.log')
        self.results_file = os.path.join('logs', 'results.json')
        
        # 使用 deque 来存储最新的2000行日志
        self.workflow_logs = deque(maxlen=2000)
        
        # 如果工作流日志文件存在，读取最后2000行
        if os.path.exists(self.workflow_log):
            with open(self.workflow_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.workflow_logs.extend(lines[-2000:])
        
        # 初始化或加载结果文件
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        else:
            self.results = {
                'last_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_tasks': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 1.0,
                'tasks': []
            }
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _save_results(self):
        """保存结果到文件"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
    
    def _setup_logger(self, name, log_file, format_str):
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5
        )
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def log_workflow(self, message):
        """记录工作流程日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}\n"
        
        # 添加到 deque
        self.workflow_logs.append(log_entry)
        
        # 写入文件
        with open(self.workflow_log, 'w', encoding='utf-8') as f:
            f.writelines(self.workflow_logs)
    
    def log_workflow_error(self, message):
        """记录工作流程错误日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - ERROR: {message}\n"
        
        # 添加到 deque
        self.workflow_logs.append(log_entry)
        
        # 写入文件
        with open(self.workflow_log, 'w', encoding='utf-8') as f:
            f.writelines(self.workflow_logs)
    
    def log_task(self, task_id, epoch, loss, accuracy):
        """记录任务日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - [TASK] - Task {task_id} - Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy*100:.2f}%\n"
        
        with open(self.task_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def get_workflow_logs(self):
        """获取工作流程日志"""
        return list(self.workflow_logs)
    
    def log_task_summary(self, task_id, final_acc):
        """记录任务的最终结果"""
        logger = self._setup_logger('task_logger', self.task_log, '%(asctime)s - [TASK] - %(message)s')
        logger.info(
            f"Task {task_id} completed - Final accuracy: {final_acc*100:.2f}%"
        )
    
    # 结果管理方法
    def save_task_result(self, task_id, accuracy, details=None):
        """保存任务结果到单个JSON文件"""
        task_data = {
            'task_id': task_id,
            'accuracy': accuracy,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': details or {}
        }
        
        # 更新任务列表
        self.results['tasks'].append(task_data)
        
        # 更新统计信息
        accuracies = [t['accuracy'] for t in self.results['tasks']]
        self.results.update({
            'last_update': task_data['timestamp'],
            'total_tasks': len(accuracies),
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'std_accuracy': (sum((x - self.results['mean_accuracy'])**2 for x in accuracies) / len(accuracies))**0.5,
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies)
        })
        
        # 保存到单个文件
        self._save_results()
    
    def get_task_result(self, task_id):
        """获取特定任务的结果"""
        task_file = os.path.join('logs', f'task_{task_id}.json')
        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_summary(self):
        """获取结果汇总"""
        return self.results 