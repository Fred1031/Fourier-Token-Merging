# Fourier Token Merging: Understanding and Capitalizing Frequency Domain for Efficient Image Generation

This paper introduces *Fourier Token Merging*, a new method for understanding and capitalizing frequency domain for efficient image generation. By introducing frequency token merging, we find that transforming the token into the frequency domain representation for clustering can better exert the ability of clustering based on the underlying redundancy after de-correlation. Through analytical and empirical studies, we demonstrate the benefits of using Fourier clustering over the original time domain clustering. We experimented fourier token merging on the stable diffusion model, and the results show up to 25\% reduction in latency without impairing image quality.

The code used for empirical experiments are available in this repository for validation and replication.

## Pre-requisites
This project was primarily developed with CUDA GPU. No CUDA compilation is required.  We have tested it on RTX 4090.


## Setup

```sh
conda create -n llm python=3.9
conda env create -f environment.yaml
conda activate llm
```

As always, activate the environment and install dependencies:
```sh
conda activate llm
```

## Datasets 
These are the datasets and domains used. We use ImageNet evaluation dataset. 
First download the ImageNet-eval dataset from ILSVRC2012 directory and store it in the `data/imagenet_val_50000` directory. 
Prepare for the the `devkit data` directory.
`cp ILSVRC2012_validation_ground_truth.txt ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt`
Run `python process.py`.
Then run `sample_5000.py` to select 5 images in each class and store them in a new separate directory. 
Extract data from sub-directory to only the directory `imagenet_val_1000flat`: `python extract_first1000.py`.
Place the resized datasets in `data/imagenet_val_1000flat_resized` using `resize.py`.


## Models
Models that we experimented are Stable Diffusion (SD) (CC-BY 4.0) models v1.5.

## Obtaining Results
The Fourier Token Merging is located in the `src` directory. The files `run_experiments_{*}.sh` are used for evaluating the methods and obatining the results. It involves generating two images with Fourier Token Merging for each class and store them in the EXP directory. It will output the generation latency and call apis to compute the similarity between the generated images and baseline obtained in the Datasets section.

