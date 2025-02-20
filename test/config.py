class TestConfig:
    def __init__(self):
        # 测试参数
        self.n_way = 5
        self.n_support = 5
        self.n_query = 15
        self.task_num = 100
        self.dataset = 'EuroSAT'
        
        # 标签集
        self.label_set = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 
            'Highway', 'Industrial', 'Pasture', 
            'PermanentCrop', 'Residential', 'River', 'SeaLake'
        ]
        
        # 输出目录
        self.output_support_dir = 'support_images/5_zkz'
        self.output_Lora_dir = 'support_images/Lora_images'
        self.output_dir = 'support_images'
        
        # 模型参数
        self.model_name = 'resnet18'
        self.freeze_backbone = False
        
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 