import os
import shutil

# source
src_root = "imagenet_val_5000"
# target
dst_root = "imagenet_val_1000flat"

os.makedirs(dst_root, exist_ok=True)

classes = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
classes10 = classes[:1000]

print("The classes", classes10)

for cls in classes10:
    cls_dir = os.path.join(src_root, cls)
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src_path = os.path.join(cls_dir, fname)
        dst_path = os.path.join(dst_root, fname)
        # dst_path = os.path.join(dst_root, f"{cls}_{fname}")
        shutil.copy(src_path, dst_path)

print(f"All images copied to `{dst_root}`。")
