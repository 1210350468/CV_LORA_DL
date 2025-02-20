import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms,utils
import matplotlib.pyplot as plt

class CombinedDataset(Dataset):
    def __init__(self,  generated_image_dir, label_set, original_labels_tensor, transform=None):
        self.generated_image_dir = generated_image_dir
        self.label_set = label_set
        self.transform = transform
        self.generated_image_paths = []
        self.generated_labels = []
        self.label_to_index = {label: idx for idx, label in enumerate(label_set)}
        # 逆映射索引到标签名
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # 从原始标签张量中提取支持集类别顺序
        support_labels = original_labels_tensor[:, 0].tolist()  # 每行第一个标签
        self.support_categories = [self.index_to_label[label] for label in support_labels]
        print("支持集类别顺序:", self.support_categories)
        
        # 按照支持集类别顺序加载生成的图片和标签
        self._load_generated_images()
        
    def _load_generated_images(self):
        if not os.path.exists(self.generated_image_dir):
            print(f"目录 {self.generated_image_dir} 不存在。")
            return
        
        filenames = sorted(os.listdir(self.generated_image_dir))
        
        # 按照支持集类别顺序加载生成的图片
        for category in self.support_categories:
            category_files = [f for f in filenames if f.startswith(category + '_')]
            category_files = sorted(category_files)
            for filename in category_files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    label_idx = self.label_to_index[category]
                    img_path = os.path.join(self.generated_image_dir, filename)
                    self.generated_image_paths.append(img_path)
                    self.generated_labels.append(label_idx)
                    
    def __len__(self):
        return len(self.generated_image_paths)
    
    def __getitem__(self, idx):
        idx_generated = idx
        img_path = self.generated_image_paths[idx_generated]
        label = self.generated_labels[idx_generated]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
def Load_Lora_image(label_set,original_labels_tensor,transform=None):
    # 标签集合
    # label_set = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    #             'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    #             'River', 'SeaLake']


    # 原始标签张量
    # original_labels_tensor = torch.tensor([
    #         [8, 8, 8, 8, 8],
    #         [5, 5, 5, 5, 5],
    #         [6, 6, 6, 6, 6],
    #         [7, 7, 7, 7, 7],
    #         [9, 9, 9, 9, 9]
    # ])

    # 图像预处理
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((224, 224)),
    #     torchvision.transforms.ToTensor(),
    # ])
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 创建合并的数据集
    combine = CombinedDataset(
        generated_image_dir="../test/support_images/Lora_images",
        label_set=label_set,
        original_labels_tensor=original_labels_tensor,
        transform=transform
    )

    # 创建数据加载器，确保不打乱数据顺序
    dataloader = DataLoader(combine, batch_size=5, shuffle=False, num_workers=4)

    # 分离原始和生成的标签张量
    generated_length = len(combine.generated_image_paths)
    all_images = []
    all_labels = []
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
    all_images_tensor = torch.cat(all_images)
    all_labels_tensor = torch.cat(all_labels)

    print("生成图片数量：",generated_length)
    # 分割标签张量
    generated_loaded_images = all_images_tensor[:generated_length]
    generated_loaded_labels = all_labels_tensor[:generated_length]

    # 调整形状
    print("调整形状前的images:",generated_loaded_images.shape)
    print("调整形状前的labels:",generated_loaded_labels.shape)
    generated_loaded_images = generated_loaded_images.view(-1, 5 , 3, 224, 224)
    generated_loaded_labels = generated_loaded_labels.view(-1, 5)

    print("调整形状后的images:",generated_loaded_images.shape)
    print("调整形状后的labels:",generated_loaded_labels.shape)
    # 合并标签张量
    Lora_generate = generated_loaded_labels
    # combined_labels = torch.cat((original_labels_tensor, Lora_generate), dim=0)

    print("原始标签张量:")
    print(original_labels_tensor)
    print("\nLora_generate 标签张量:")
    print(Lora_generate)
    # print("\n合并后的标签张量:")
    # print(combined_labels)

    # # 可视化一批数据
    # def visualize_batch(images, labels, label_set):
    #     grid = utils.make_grid(images)
    #     npimg = grid.numpy().transpose((1, 2, 0))
    #     plt.imshow(npimg)
    #     titles = [label_set[label] for label in labels]
    #     plt.title(", ".join(titles))
    #     plt.axis('off')
    #     plt.show()

    # for i in range(generated_loaded_images.size(0)):
    #     images = generated_loaded_images[i]
    #     labels = generated_loaded_labels[i]
    #     print("label ")
    #     print(labels)
    #     visualize_batch(images,labels,label_set)
    return generated_loaded_images,generated_loaded_labels

# for images, labels in dataloader:
#     print("第一批次的标签:")
#     print(labels)
#     print(images.shape)
#     visualize_batch(images, labels, label_set)