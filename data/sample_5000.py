import os
import random
import shutil
from collections import defaultdict

# 原始 ImageNet 验证集所在目录，每个类别存放在一个子文件夹中
val_dir = "imagenet_val_by_class"  # 修改为实际路径
# 输出目录，用于存放抽取后的 5,000 张图像
output_dir = "imagenet_val_5000"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个类别文件夹
for cls in os.listdir(val_dir):
    cls_path = os.path.join(val_dir, cls)
    if os.path.isdir(cls_path):
        # 列出该类别下所有图像
        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 随机抽取 5 张图像（若该类别图像少于 5 张，则取全部）
        selected = random.sample(images, min(5, len(images)))
        
        # 在输出目录中为该类别创建文件夹
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)
        
        # 将选中的图像复制到输出目录中
        for img in selected:
            src = os.path.join(cls_path, img)
            dst = os.path.join(cls_output, img)
            shutil.copy(src, dst)

print("抽取完成，5,000 张类别平衡的验证样本已存放在:", output_dir)
