import os
import argparse
import torch, src
import csv
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument("--use_tomesd",   type=int,   default=1,   help="switch for ToMe (0/1)")
parser.add_argument("--tomesd_ratio", type=float, default=0.5, help="ToMe merge ratio")
parser.add_argument("--use_dft",      type=int,   default=0,   help="switch for DFT truncation (0/1)")
parser.add_argument("--trunc_ratio",  type=float, default=1.0, help="DFT truncation ratio (0.0–1.0)")
parser.add_argument("--output_dir",   type=str,   default="generated_images", help="directory to save generated images")
parser.add_argument("--high_weight",  type=float,   default=1, help="weight for low frequency")
parser.add_argument("--low_weight",   type=float,   default=1, help="weight for high frequency")

### adaptive truncation
parser.add_argument("--adaptive_trunc", action="store_true", help="Enable adaptive truncation")
parser.add_argument("--adaptive_basis", type=str, default="timestep", choices=["timestep", "layer"], help="Use timestep or layer as adaptive basis")
parser.add_argument("--adaptive_direction", type=str, default="increasing", choices=["increasing", "decreasing"], help="Direction of truncation change")
parser.add_argument("--trunc_ratio_min", type=float, default=0.3, help="Minimum truncation ratio for adaptive mode")
parser.add_argument("--trunc_ratio_max", type=float, default=0.9, help="Maximum truncation ratio for adaptive mode")
parser.add_argument("--max_timestep", type=int, default=999, help="Maximum timestep (for normalization)")
parser.add_argument("--total_layers", type=int, default=16, help="Total number of layers (for layer-based adaptation)")



parser.add_argument("--debug",        type=bool,   default=False, help="switch to debug mode")
args = parser.parse_args()


use_tomesd    = bool(args.use_tomesd)
tomesd_ratio  = args.tomesd_ratio
use_dft       = bool(args.use_dft)
trunc_ratio   = args.trunc_ratio
output_dir    = args.output_dir
high_weight   = args.high_weight
low_weight    = args.low_weight

# ----- Step 1:  ImageNet-1k classification list -----
def load_imagenet_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

imagenet_classes = load_imagenet_classes("imagenet_classes.txt")
assert len(imagenet_classes) == 1000, "The number of classes should be 1000."

# ----- Step 2: generate images using Stable Diffusion v1.5 -----
os.makedirs(output_dir, exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

pipe = pipe.to("cuda")  
pipe.enable_attention_slicing()  

if use_tomesd:
    src.apply_patch(
        pipe,
        ratio=tomesd_ratio,
        use_dft=use_dft,
        trunc_ratio=trunc_ratio,
        high_weight=high_weight,
        low_weight=low_weight,

        adaptive_trunc=args.adaptive_trunc,
        adaptive_basis=args.adaptive_basis,
        adaptive_direction=args.adaptive_direction,
        trunc_ratio_min=args.trunc_ratio_min,
        trunc_ratio_max=args.trunc_ratio_max,
        max_timestep=args.max_timestep,
        total_layers=args.total_layers,
    )
else:
    try:
        src.remove_patch(pipe)
    except:
        pass

num_inference_steps = 50
guidance_scale = 7.5

trial = 0

total_generation_time = 0.0

image_count = 0
for class_name in tqdm(imagenet_classes, desc="generating images"):
    prompt = f"a photo of a {class_name}"
    for i in range(2):  # 2 images per class
        start = timer()

        output = pipe(prompt, height=512, width=512, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        total_generation_time += timer() - start
        image = output["images"][0]
        file_name = f"{class_name.replace(' ', '_')}_{i}.png"
        image.save(os.path.join(output_dir, file_name))
        image_count += 1

    if args.debug and image_count >= 2:
        break
end = timer()
print(f"Latency: {total_generation_time / image_count:.2f} s")

with open(os.path.join("./", "generation_time.txt"), "a") as f:
    writer = csv.writer(f)
    writer.writerow([f"{total_generation_time / image_count:.2f}"])

print('latency: ', total_generation_time / image_count)

print(f"Generating {image_count} images, stored in {output_dir}")

