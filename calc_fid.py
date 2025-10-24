#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torch_fidelity import calculate_metrics

class FlatFolderDatasetUInt8(Dataset):
    def __init__(self, folder: str, size=(299, 299)):
        self.paths = [
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if fn.lower().endswith(('.jpg','.jpeg','.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        print(idx)
        img = read_image(self.paths[idx])  # uint8 [C,H,W]
        return self.transform(img)         # uint8 [C,299,299]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",  type=str, default="generated_images/", help="generated images directory")
    parser.add_argument("--real_dir", type=str, default="data/imagenet_val_1000flat_resized", help="real images directory")
    args = parser.parse_args()

    gen_ds  = FlatFolderDatasetUInt8(args.gen_dir,  size=(512,512))
    real_ds = FlatFolderDatasetUInt8(args.real_dir, size=(512,512))

    '''
    metrics = calculate_metrics(
        #input1=gen_ds,
        #input2=real_ds,
        input1=args.gen_dir,
        input2=args.real_dir,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=False,
        kid=False,
        batch_size=16,
        num_workers=4,
        resize=True,
        crop=True,
    )
    '''

    metrics = calculate_metrics(
    #input1="generated_images",
    #input1="gen_ut0",
    input1=args.gen_dir,
    #input2="data/imagenet_val_10flat_resized",
    #input2="data/imagenet_val_1000flat_resized",
    input2=args.real_dir,
    cuda=True,
    fid=True,
    isc=False,
    kid=False,
    batch_size=16,
    num_workers=4,
    resize=True,   # auto resize 
    crop=True      # auto center crop
)

    fid = metrics["frechet_inception_distance"]
    print(f"{fid:.4f}")

if __name__ == "__main__":
    main()
