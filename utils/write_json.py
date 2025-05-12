import os
import json

def convert_dataset_to_json(dataset_dir, output_json):
    dataset_dict = []
    label_mapping = {}  # 类别名称到标签编号的映射
    current_label = 0

    for class_name in sorted(os.listdir(dataset_dir)):  # 按字典顺序读取类别
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            if class_name not in label_mapping:
                label_mapping[class_name] = current_label
                current_label += 1
            
            label = label_mapping[class_name]
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    dataset_dict.append({
                        "image_path": img_path,
                        "label": label,
                        "label_name": class_name  # 添加类别名称
                    })

    # 保存为 JSON 文件
    with open(output_json, 'w') as json_file:
        json.dump(dataset_dict, json_file, indent=4)

    # 输出标签映射，用于检查
    print("Label Mapping:", label_mapping)

# 调用函数进行转换，确保生成的 JSON 包含所有类别及其对应的标签
# convert_dataset_to_json('path_to_dataset', 'output_dataset.json')


# 调用函数进行转换
# convert_dataset_to_json('/root/autodl-tmp/all_starnight/dataset/datafsl/source/EuroSAT_add/images', 'EuroSAT.json')
convert_dataset_to_json('/root/autodl-tmp/all_starnight/dataset/datafsl/source/miniImagenet/train', 'miniImagenet_train.json')
convert_dataset_to_json('/root/autodl-tmp/all_starnight/dataset/datafsl/source/miniImagenet/val', 'miniImagenet_val.json')
convert_dataset_to_json('/root/autodl-tmp/all_starnight/dataset/datafsl/source/miniImagenet/test', 'miniImagenet_test.json')