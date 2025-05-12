import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import ftfy
import regex as re
from typing import Union, List
try:
    import open_clip
except ImportError:
    print("Installing open_clip...")
    os.system('pip install open-clip-torch')
    import open_clip

class CLIPSimilarityCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # 使用open_clip加载模型
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model = self.model.to(device)
        self.model.eval()
        
    def load_image(self, image_path):
        """加载并预处理图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def compute_similarity(self, image1_path, image2_path):
        """计算两张图片的相似度"""
        # 加载图片
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)
        
        if image1 is None or image2 is None:
            return 0.0
        
        # 提取特征
        with torch.no_grad():
            feat1 = self.model.encode_image(image1)
            feat2 = self.model.encode_image(image2)
            
            # 归一化特征
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
        return similarity.item()
    
    def compute_batch_similarity(self, support_dir, lora_dir, save_path=None):
        """计算两个目录下同类别图片的相似度矩阵"""
        # 检查目录是否存在
        if not os.path.exists(support_dir):
            raise ValueError(f"Support directory not found: {support_dir}")
        if not os.path.exists(lora_dir):
            raise ValueError(f"LoRA directory not found: {lora_dir}")
        
        # 获取所有图片路径并按前缀分组
        def get_prefix(filename):
            return filename.split('_')[0]  # 假设前缀用下划线分隔
        
        support_images = sorted([f for f in os.listdir(support_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        lora_images = sorted([f for f in os.listdir(lora_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 按前缀分组
        support_by_prefix = {}
        lora_by_prefix = {}
        
        for img in support_images:
            prefix = get_prefix(img)
            if prefix not in support_by_prefix:
                support_by_prefix[prefix] = []
            support_by_prefix[prefix].append(img)
        
        for img in lora_images:
            prefix = get_prefix(img)
            if prefix not in lora_by_prefix:
                lora_by_prefix[prefix] = []
            lora_by_prefix[prefix].append(img)
        
        # 找出共同的前缀
        common_prefixes = set(support_by_prefix.keys()) & set(lora_by_prefix.keys())
        print(f"Found {len(common_prefixes)} common categories: {sorted(common_prefixes)}")
        
        # 为每个类别计算相似度矩阵
        results = {}
        for prefix in common_prefixes:
            support_class_images = support_by_prefix[prefix]
            lora_class_images = lora_by_prefix[prefix]
            
            print(f"\nProcessing category '{prefix}':")
            print(f"Support images: {len(support_class_images)}")
            print(f"LoRA images: {len(lora_class_images)}")
            
            # 初始化相似度矩阵
            similarity_matrix = np.zeros((len(support_class_images), len(lora_class_images)))
            
            # 计算相似度
            for i, support_img in enumerate(tqdm(support_class_images, desc=f"Processing {prefix}")):
                for j, lora_img in enumerate(lora_class_images):
                    support_path = os.path.join(support_dir, support_img)
                    lora_path = os.path.join(lora_dir, lora_img)
                    similarity = self.compute_similarity(support_path, lora_path)
                    similarity_matrix[i, j] = similarity
            
            results[prefix] = {
                'matrix': similarity_matrix,
                'support_images': support_class_images,
                'lora_images': lora_class_images
            }
            
            # 为每个类别生成单独的热力图
            if save_path:
                plt.figure(figsize=(10, 6))
                sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis')
                plt.xlabel('LoRA Images')
                plt.ylabel('Support Images')
                plt.title(f'CLIP Similarity Matrix - Category: {prefix}')
                
                # 生成类别特定的保存路径
                category_save_path = os.path.join(
                    os.path.dirname(save_path),
                    f'similarity_matrix_{prefix}.png'
                )
                os.makedirs(os.path.dirname(category_save_path), exist_ok=True)
                plt.savefig(category_save_path)
                plt.close()
        
        # 保存统计信息
        if save_path:
            stats_path = os.path.splitext(save_path)[0] + '_stats.txt'
            with open(stats_path, 'w') as f:
                f.write("Similarity Statistics by Category:\n\n")
                
                for prefix in common_prefixes:
                    matrix = results[prefix]['matrix']
                    f.write(f"\nCategory: {prefix}\n")
                    f.write(f"Average similarity: {np.mean(matrix):.4f}\n")
                    f.write(f"Std similarity: {np.std(matrix):.4f}\n")
                    f.write(f"Min similarity: {np.min(matrix):.4f}\n")
                    f.write(f"Max similarity: {np.max(matrix):.4f}\n")
                    
                    # 找出最相似和最不相似的图片对
                    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
                    min_idx = np.unravel_index(np.argmin(matrix), matrix.shape)
                    
                    f.write("\nMost similar pair:\n")
                    f.write(f"Support: {results[prefix]['support_images'][max_idx[0]]}\n")
                    f.write(f"LoRA: {results[prefix]['lora_images'][max_idx[1]]}\n")
                    f.write(f"Similarity: {matrix[max_idx]:.4f}\n")
                    
                    f.write("\nLeast similar pair:\n")
                    f.write(f"Support: {results[prefix]['support_images'][min_idx[0]]}\n")
                    f.write(f"LoRA: {results[prefix]['lora_images'][min_idx[1]]}\n")
                    f.write(f"Similarity: {matrix[min_idx]:.4f}\n")
                    f.write("\n" + "="*50 + "\n")
        
        return results

def main():
    try:
        # 使用示例
        calculator = CLIPSimilarityCalculator()
        
        # 设置图片目录
        support_dir = "../test/support_images/5_zkz"
        lora_dir = "../test/support_images/Lora_images"
        save_path = "../test/support_images/similarity_matrix.png"
        
        print(f"Support directory: {os.path.abspath(support_dir)}")
        print(f"LoRA directory: {os.path.abspath(lora_dir)}")
        
        # 计算相似度矩阵并保存结果
        results = calculator.compute_batch_similarity(support_dir, lora_dir, save_path)
        
        # 打印每个类别的平均相似度
        print("\nAverage similarities by category:")
        for prefix, data in results.items():
            matrix = data['matrix']
            print(f"{prefix}:")
            print(f"  Average: {np.mean(matrix):.4f}")
            print(f"  Min: {np.min(matrix):.4f}")
            print(f"  Max: {np.max(matrix):.4f}")
            print(f"  Std: {np.std(matrix):.4f}")
        
        # 计算所有类别的总体平均相似度
        all_similarities = np.concatenate([data['matrix'].flatten() for data in results.values()])
        print(f"\nOverall statistics:")
        print(f"Average similarity across all categories: {np.mean(all_similarities):.4f}")
        print(f"Standard deviation: {np.std(all_similarities):.4f}")
        print(f"Min similarity: {np.min(all_similarities):.4f}")
        print(f"Max similarity: {np.max(all_similarities):.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 