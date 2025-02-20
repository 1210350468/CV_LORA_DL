from flask import Flask, render_template, jsonify, request
from Test_Metalearn_Finetune import TestManager
import os
import signal
import atexit
from threading import Thread
import socket
import psutil
import logging
from logging.handlers import RotatingFileHandler
import datetime
import json
import traceback

app = Flask(__name__)

# 全局的test manager实例
test_manager = None
test_thread = None

def find_free_port(start_port=5000, max_port=5100):
    """查找可用的端口"""
    for port in range(start_port, max_port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")

def kill_existing_flask():
    """终止已存在的Flask进程"""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in cmdline[0] and 'app.py' in ' '.join(cmdline) and proc.info['pid'] != current_pid:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def kill_process_using_port(port):
    """终止占用指定端口的进程"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # 获取进程的所有连接
            connections = proc.connections()
            for conn in connections:
                if hasattr(conn, 'laddr') and conn.laddr.port == port:
                    try:
                        proc.kill()
                        print(f"Killed process {proc.pid} using port {port}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def cleanup():
    """清理函数，确保程序退出时能够正确清理资源"""
    global test_thread
    if test_thread and test_thread.is_alive():
        # 在这里可以添加中断测试的逻辑
        pass

# 注册清理函数
atexit.register(cleanup)

# 处理SIGTERM信号
def handle_sigterm(signum, frame):
    cleanup()
    os._exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_test', methods=['POST'])
def start_test():
    """启动测试"""
    global test_manager, test_thread
    
    try:
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 初始化测试管理器
        if test_manager is not None:
            test_manager.clean_directories()
        
        test_manager = TestManager()
        
        # 启动测试线程
        test_thread = Thread(target=test_manager.run_test)
        test_thread.daemon = True
        test_thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Test started successfully',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'n_way': test_manager.config.n_way,
                'n_support': test_manager.config.n_support,
                'dataset': test_manager.config.dataset
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Failed to start test. Please check the configuration.',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/get_status')
def get_status():
    if test_manager is None:
        return jsonify({
            'state': 'not_started',
            'current_task': 0,
            'total_tasks': 0,
            'current_epoch': 0,
            'total_epochs': 100,
            'accuracy': 0.0,
            'loss': 0.0,
            'log_messages': []
        })
    
    try:
        status = test_manager.status
        # 添加loss到状态中
        status['loss'] = status.get('task_loss', 0.0)
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'state': 'error',
            'error_message': str(e),
            'log_messages': test_manager.status.get('log_messages', [])
        })

@app.route('/get_logs')
def get_logs():
    try:
        if test_manager is None:
            return jsonify({'logs': []})
            
        # 从workflow日志文件读取最新的日志
        workflow_log_path = os.path.join('logs', 'workflow.log')
        if os.path.exists(workflow_log_path):
            with open(workflow_log_path, 'r', encoding='utf-8') as f:
                workflow_logs = f.readlines()
                # 清理日志行，移除多余的换行符
                logs = [log.strip() for log in workflow_logs[-100:] if log.strip()]
                return jsonify({'logs': logs})
        return jsonify({'logs': []})
    except Exception as e:
        app.logger.error(f"Error getting workflow logs: {str(e)}")
        return jsonify({'error': str(e), 'logs': []})

@app.route('/get_task_logs')
def get_task_logs():
    try:
        if test_manager is None:
            return jsonify({'logs': []})
            
        # 从task输出日志文件读取最新的日志
        task_log_path = os.path.join('logs', 'task_output.log')
        if os.path.exists(task_log_path):
            with open(task_log_path, 'r', encoding='utf-8') as f:
                task_logs = f.readlines()
                # 清理日志行，移除多余的换行符
                logs = [log.strip() for log in task_logs[-100:] if log.strip()]
                return jsonify({'logs': logs})
        return jsonify({'logs': []})
    except Exception as e:
        app.logger.error(f"Error getting task logs: {str(e)}")
        return jsonify({'error': str(e), 'logs': []})

@app.route('/get_task_results')
def get_task_results():
    try:
        if test_manager is None:
            return jsonify({'results': None})
            
        # 从results.json读取结果
        results_path = os.path.join('logs', 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                return jsonify({
                    'summary': results,
                    'current_task': test_manager.status.get('current_task', 0)
                })
        return jsonify({
            'summary': {
                'total_tasks': 0,
                'mean_accuracy': 0.0,
                'best_accuracy': 0.0,
                'tasks': []
            },
            'current_task': 0
        })
    except Exception as e:
        app.logger.error(f"Error getting task results: {str(e)}")
        return jsonify({'error': str(e), 'results': None})

@app.route('/get_task_details/<int:task_id>')
def get_task_details(task_id):
    if test_manager is None:
        return jsonify({'result': None})
    return jsonify({
        'result': test_manager.result_manager.get_task_result(task_id)
    })

@app.route('/get_all_logs')
def get_all_logs():
    """获取所有日志文件的内容"""
    try:
        logs_data = {
            'workflow': [],
            'task': [],
            'results': None
        }
        
        # 读取workflow日志
        workflow_log_path = os.path.join('logs', 'workflow.log')
        if os.path.exists(workflow_log_path):
            with open(workflow_log_path, 'r', encoding='utf-8') as f:
                logs_data['workflow'] = [line.strip() for line in f.readlines() if line.strip()]
        
        # 读取task日志
        task_log_path = os.path.join('logs', 'task_output.log')
        if os.path.exists(task_log_path):
            with open(task_log_path, 'r', encoding='utf-8') as f:
                logs_data['task'] = [line.strip() for line in f.readlines() if line.strip()]
        
        # 读取结果文件
        results_path = os.path.join('logs', 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                logs_data['results'] = json.load(f)
        
        return jsonify(logs_data)
    except Exception as e:
        app.logger.error(f"Error getting all logs: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_test_checklist():
    """生成测试清单"""
    checklist = [
        "1. 检查日志目录是否正确创建",
        "2. 验证日志文件格式是否正确",
        "3. 确认前端能够正确显示工作流程日志",
        "4. 确认前端能够正确显示任务进度日志",
        "5. 验证指标数据更新是否正常",
        "6. 检查任务历史记录是否正确保存",
        "7. 验证错误处理机制是否正常工作",
        "8. 确认清理功能是否正常运行",
        "9. 检查 Git 同步是否成功",
        "10. 验证开发日志是否正确记录"
    ]
    
    with open('logs/test_checklist.txt', 'w') as f:
        f.write(f"Test Checklist - {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write("="*50 + "\n")
        for item in checklist:
            f.write(f"[ ] {item}\n")

if __name__ == '__main__':
    try:
        port = 7860
        print("="*50)
        print(f"Server starting on port {port}")
        print(f"Please access the dashboard using AutoDL's service URL on port {port}")
        print("="*50)
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        cleanup()