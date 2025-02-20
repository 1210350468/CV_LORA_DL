
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.Metastage_Dataloader import get_meta_dataloader
from dataloader.Pretrain_Dataloader import get_pretrain_dataloader
def test_dataloader(dataloader, phase):
    print(f"Testing {phase} dataloader...")

    for i, data in enumerate(dataloader):
        task_images, task_labels = data
        print(f"Batch {i + 1}:")
        print(f"  Task images shape: {task_images.shape}")  # 应该输出 [n_way, n_support + n_query, 3, size, size]
        print(f"  Task labels: {task_labels}")  # 输出每个类别的标签
        print(f"label shape: {len(task_labels)}")
        # 测试只运行一次 batch，确保形状正常
        break

    print(f"{phase} dataloader test completed.\n")

if __name__ == "__main__":
    # 替换为你实际的 JSON 文件路径
    pretrain_json_file = './utils/miniImage_train.json'
    metalearning_json_file = './utils/EuroSAT.json'
    metatest_json_file = './utils/EuroSAT.json'
    n_way = 5
    n_support = 5
    n_query = 15
    # 测试 Pretrain
    # pretrain_loader = get_pretrain_dataloader(batch_size=32, json_file=pretrain_json_file)
    # test_dataloader(pretrain_loader, "Pretrain")
    # input()
    # 测试 Metalearning
    metalearning_loader = get_meta_dataloader(num_classes=n_way,num_support=n_support,num_query=n_query, json_file=metalearning_json_file)
    test_dataloader(metalearning_loader, "Metalearning")
    
    # 测试 Metatest
    # metatest_loader = get_meta_dataloader(batch_size=32,num_classes=n_way,num_support=n_support,num_query=n_query, json_file=metalearning_json_file)
    # test_dataloader(metatest_loader, "Metatest")
