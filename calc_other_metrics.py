#!/usr/bin/env python3
import os
import argparse
import torch
import lpips
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def load_and_preprocess(image_path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1] for LPIPS
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",  type=str, required=True, help="Generated image folder")
    parser.add_argument("--real_dir", type=str, required=True, help="Ground-truth image folder")
    args = parser.parse_args()

    gen_files = sorted([f for f in os.listdir(args.gen_dir) if is_image_file(f)])
    real_files = sorted([f for f in os.listdir(args.real_dir) if is_image_file(f)])
    common_files = sorted(set(gen_files).intersection(set(real_files)))

    if not common_files:
        raise ValueError("No matching image filenames found between the two folders.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    lpips_scores = []
    ms_ssim_scores = []

    for name in tqdm(common_files, desc="Evaluating"):
        img_gen  = load_and_preprocess(os.path.join(args.gen_dir, name)).to(device)
        img_real = load_and_preprocess(os.path.join(args.real_dir, name)).to(device)

        with torch.no_grad():
            lpips_score = lpips_model(img_gen, img_real).item()
            ms_ssim_score = ms_ssim(img_gen, img_real, data_range=2.0).item()

        lpips_scores.append(lpips_score)
        ms_ssim_scores.append(ms_ssim_score)

    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    avg_msssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
    print(f"\nLPIPS (↓):   {avg_lpips:.4f}")
    print(f"MS-SSIM (↑): {avg_msssim:.4f}")

if __name__ == "__main__":
    main()
