import os
def trainlora():
    # 保存当前的工作目录
    original_dir = os.getcwd()
    print(f"当前的工作目录: {original_dir}")
    # 定义脚本路径和工作目录
    script_path = "/root/autodl-tmp/all_starnight/lora-scripts/train_by_toml.sh"  # 你的 .sh 脚本的完整路径
    work_dir = os.path.dirname(script_path)  # 获取脚本所在的目录
    print(f"脚本路径: {script_path}")
    try:
        # 切换到脚本所在的目录并执行脚本
        os.chdir(work_dir)
        os.system(f"bash {script_path}")
    finally:
        # 切换回原来的工作目录
        os.chdir(original_dir)
        print(f"已返回到原来的工作目录: {original_dir}")