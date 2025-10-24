import os
import shutil
import scipy.io

val_dir = "imagenet_val_50000"                  
devkit_dir = "ILSVRC2012_devkit_t12/data"        # devkit data directory
out_dir = "imagenet_val_by_class"                # output directory

os.makedirs(out_dir, exist_ok=True)

# --- 1. Read ground truth labels ---
gt_path = os.path.join(devkit_dir, "ILSVRC2012_validation_ground_truth.txt")
with open(gt_path, "r") as f:
    gt_labels = [int(line.strip()) for line in f]  

#print(len(gt_labels))

# --- 2. Read meta.mat，get the class_id -> wnid mapping---
meta = scipy.io.loadmat(os.path.join(devkit_dir, "meta.mat"))
synsets = meta["synsets"]  
print(len(synsets))
classid2wnid = {}
for s in synsets:
    wnid = str(s["WNID"][0])                  # e.g. 'n01440764'
    class_id = int(s["ILSVRC2012_ID"][0][0])  # 1..1000
    classid2wnid[class_id] = wnid

print(classid2wnid)

# --- 3. sort according to file name and store them to directory ---
files = sorted(os.listdir(val_dir))
#print(files)
assert len(files) == len(gt_labels), "Number of files and labels should match."

for fname, cls in zip(files, gt_labels):
    #print(cls)
    wnid = classid2wnid[cls]
    dst_cls_dir = os.path.join(out_dir, wnid)
    os.makedirs(dst_cls_dir, exist_ok=True)
    shutil.copy(
        os.path.join(val_dir, fname),
        os.path.join(dst_cls_dir, fname)
    )

print("Done. ", out_dir)
