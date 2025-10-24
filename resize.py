from PIL import Image
import glob, os

def batch_resize(src_folder, dst_folder, size=(512,512)):
    os.makedirs(dst_folder, exist_ok=True)
    for fn in glob.glob(f"{src_folder}/*.*"):
        img = Image.open(fn).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        basename = os.path.basename(fn)
        img.save(os.path.join(dst_folder, basename))


batch_resize("data/imagenet_val_1000flat","data/imagenet_val_1000flat_resized")
