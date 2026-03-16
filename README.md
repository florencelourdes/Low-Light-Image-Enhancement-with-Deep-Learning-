# Low-Light Image Enhancement with Deep Learning  
## Full Pipeline (GPU Version)

This repository contains the full implementation for my **low-light image enhancement project**. The goal of the project is to explore how different deep learning architectures improve low-light images and evaluate their impact on downstream tasks.

The project compares multiple CNN architectures for enhancement, evaluates them using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**, and explores a downstream experiment using a **binary classifier on raw vs enhanced images**.

The codebase allows the entire experiment pipeline to be run automatically or individual experiments to be executed separately.

---

# Project Overview

Low-light imaging is a common problem in computer vision applications such as:

- surveillance systems  
- robotics and autonomous vehicles  
- mobile photography  
- nighttime object detection  

Images captured in poor lighting conditions often suffer from:

- low contrast  
- high noise  
- reduced dynamic range  
- loss of detail  

These issues negatively affect both **human perception** and **machine learning models**.

This project investigates how **deep learning-based enhancement models** can restore useful information from low-light images.

I compare several neural network architectures:

- Shallow CNN
- Deep CNN
- U-Net
- Residual U-Net (ResUNet)

I also analyze:

- RGB vs grayscale inputs
- quantitative metrics (PSNR / SSIM)
- qualitative image improvements
- downstream classification behavior

---

# Project Scope

The full experimental pipeline includes the following components.

## Enhancement Model Experiments

### 1. Shallow CNN (RGB)
Baseline convolutional network for low-light enhancement.

### 2. Deep CNN (RGB)
A deeper architecture to test whether network depth improves restoration quality.

### 3. U-Net (RGB)
Encoder-decoder architecture with skip connections that preserve spatial detail.

### 4. Residual U-Net (ResUNet) (RGB)
An extension of U-Net that introduces residual connections to improve gradient flow and training stability.

### 5. Deep CNN (Grayscale)
Tests whether enhancement performance is dependent on color channels.

### 6. U-Net (Grayscale)
Evaluates whether a strong architecture can perform well using only luminance information.

---

## Downstream Classification Experiment

In addition to enhancement experiments, the project includes a **binary image classifier** used to analyze how enhancement affects the visual distribution of images.

The classifier is trained to distinguish between:

- **Raw low-light images (ExDark)**
- **Enhanced images produced by the trained enhancement models**

### Purpose

This experiment investigates whether enhancement significantly changes the visual characteristics of images in a way that a machine learning model can detect.

### Pipeline

1. Enhance the ExDark dataset using a trained enhancement model.
2. Build a dataset containing:
   - raw ExDark images
   - enhanced versions of the same images
3. Train a **binary CNN classifier** to distinguish between the two classes.
4. Evaluate classifier performance using validation and test metrics.

This extension provides insight into how enhancement transformations alter image distributions and how those changes may affect downstream computer vision tasks.

---

# Evaluation Metrics

Enhancement models are evaluated using two standard image restoration metrics.

## PSNR (Peak Signal-to-Noise Ratio)

PSNR measures the ratio between signal and reconstruction error.

Higher PSNR indicates better reconstruction quality.

## SSIM (Structural Similarity Index)

SSIM measures perceived image quality based on:

- luminance
- contrast
- structural similarity

SSIM is generally more aligned with **human visual perception**.

---

# Downstream Experiment (Extension)

To explore how enhancement affects downstream tasks, I trained a **binary classifier** to distinguish:

- raw low-light ExDark images
- enhanced ExDark images

This experiment investigates whether enhancement significantly changes the visual distribution of images in a way that machine learning models can detect.

---

# Datasets Used

## LOL Dataset

Paired low-light and normal-light images used for supervised training.

Dataset structure:

```text
LOL/
 ├── our485/
 │   ├── low/
 │   └── high/
 └── eval15/
     ├── low/
     └── high/
```

---

## LoLI-Street Dataset

Additional paired dataset used for training and validation.

Dataset structure:

```text
LoLI-Street/
 ├── Train/
 │   ├── low/
 │   └── high/
 ├── Val/
 │   ├── low/
 │   └── high/
 └── Test/
```

---

## ExDark Dataset

Real-world low-light dataset used for qualitative evaluation and downstream experiments.

Unlike LOL and LoLI-Street, ExDark does not provide paired ground-truth images.

It is primarily used to demonstrate real-world enhancement results.

---

# Hardware Used

The results in the `runs/` folder were generated using:

- **Date:** March 2026  
- **GPU:** NVIDIA GeForce RTX 5070 Ti  
- **CUDA:** 12.8  
- **Framework:** PyTorch  
- **Mixed Precision:** Enabled (AMP)

All metrics and outputs included in the repository were produced using this hardware configuration.

---

# Important Note About ResUNet RGB

During the initial full pipeline run, the **ResUNet RGB training produced NaN metrics**.

This instability occurred due to a combination of:

- mixed precision training
- aggressive training configuration

To resolve this:

1. The ResUNet RGB model was retrained with stabilized settings.
2. The corrected run is stored in:

```
runs/resunet_rgb_retry
```

This folder contains the successful ResUNet results.

The retry configuration produced valid PSNR and SSIM metrics and was used for the final evaluation.

The retry results were integrated into the full project results folder:

```
runs/full_project/enhance_resunet_rgb_retry
```

### Changes made to stabilize ResUNet

- reduced learning rate
- gradient clipping
- improved numerical stability checks
- selective AMP handling

---

# Repository Structure

```text
project_root/
│
├── config.py
│
├── scripts/
│   ├── run_full_project.py
│   ├── train_enhancement.py
│   ├── evaluate_enhancement.py
│   ├── enhance_exdark.py
│   ├── train_classifier.py
│   ├── train_binary_classifier.py
│   ├── build_binary_classifier_dataset.py
│   └── export_sample_pairs.py
│
├── runs/
│   ├── full_project/
│   ├── resunet_rgb_retry/
│   └── classifier_binary/
│
└── Datasets/
    ├── LOL/
    ├── LoLI-Street/
    └── ExDark/
```

---

# Important Note About Large Files

The **Datasets** and **runs** folders are **not included in this repository** because they are too large to store on GitHub.

If you want to reproduce the experiments, you must download the datasets separately and place them in the correct folder structure.

---

# Step 1: Download the Datasets

Download the datasets from Kaggle:

- **LOL Dataset**  
  https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

- **LoLI-Street Dataset**  
  https://www.kaggle.com/datasets/krishnanshverma/loli-street

- **ExDark Dataset**  
  https://www.kaggle.com/datasets/ankitshah009/exdark

After downloading, place them inside a folder named:

```
Datasets/
```

with the following structure:

```text
Datasets/
 ├── LOL/
 │   ├── our485/
 │   │   ├── low/
 │   │   └── high/
 │   └── eval15/
 │       ├── low/
 │       └── high/
 │
 ├── LoLI-Street/
 │   ├── Train/
 │   │   ├── low/
 │   │   └── high/
 │   ├── Val/
 │   │   ├── low/
 │   │   └── high/
 │   └── Test/
 │
 └── ExDark/
```

---

# Step 2: Configure Dataset Paths

Open `config.py` and update the dataset paths to match where the datasets are stored on your machine.

Example:

```python
LOL_PATH = r"C:/datasets/LOL"
LOLI_PATH = r"C:/datasets/LoLI-Street"
EXDARK_PATH = r"C:/datasets/ExDark"
```

---

# Step 3: Generate the `runs/` Folder

The **runs folder is automatically created** when the training scripts are executed.

After running the full pipeline:

```bash
python scripts/run_full_project.py
```

the project will generate a directory structure similar to:

```text
runs/
 ├── full_project/
 ├── enhance_shallow_rgb/
 ├── enhance_unet_rgb/
 └── classifier_binary/
```

These folders contain:

- trained model checkpoints  
- evaluation metrics  
- training plots  
- enhanced image outputs  

Because these outputs can be several gigabytes, they are **not stored in the GitHub repository**.

---

# Install Dependencies

Install the CUDA-enabled PyTorch build.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install remaining dependencies:

```bash
pip install pillow numpy scikit-image matplotlib tqdm
```

---

# Running the Full Project

After configuring dataset paths, the entire experiment pipeline can be executed with a single command.

```bash
python scripts/run_full_project.py
```

This script runs the complete pipeline:

1. Train all enhancement models
2. Evaluate PSNR / SSIM
3. Export sample visualization pairs
4. Enhance ExDark dataset
5. Train downstream classifier experiments

All outputs are saved to:

```
runs/full_project
```

---

# Running Individual Experiments

## Train Enhancement Models

```bash
python scripts/train_enhancement.py --model shallow --input_mode rgb --save_dir runs/enhance_shallow_rgb

python scripts/train_enhancement.py --model deep --input_mode rgb --save_dir runs/enhance_deep_rgb

python scripts/train_enhancement.py --model unet --input_mode rgb --save_dir runs/enhance_unet_rgb

python scripts/train_enhancement.py --model resunet --input_mode rgb --save_dir runs/enhance_resunet_rgb

python scripts/train_enhancement.py --model deep --input_mode gray --save_dir runs/enhance_deep_gray

python scripts/train_enhancement.py --model unet --input_mode gray --save_dir runs/enhance_unet_gray
```

---

## Evaluate Enhancement Results

```bash
python scripts/evaluate_enhancement.py \
--checkpoint runs/full_project/enhance_resunet_rgb_retry/checkpoints/best.pt \
--model resunet \
--save_dir runs/eval_resunet_rgb
```

---

## Enhance ExDark Images

```bash
python scripts/enhance_exdark.py \
--checkpoint runs/full_project/enhance_resunet_rgb_retry/checkpoints/best.pt \
--model resunet \
--output_root runs/exdark_enhanced
```

---

## Build Binary Classifier Dataset

```bash
python scripts/build_binary_classifier_dataset.py \
--raw_root Datasets/ExDark \
--enhanced_root runs/exdark_enhanced \
--out_root runs/classifier_binary_dataset
```

---

## Train Binary Classifier

```bash
python scripts/train_binary_classifier.py \
--data_root runs/classifier_binary_dataset \
--epochs 10 \
--save_dir runs/classifier_binary
```

---

# Useful Scripts

**run_full_project.py**  
Runs the entire experiment pipeline.

**train_enhancement.py**  
Trains enhancement models.

**evaluate_enhancement.py**  
Computes PSNR and SSIM metrics.

**enhance_exdark.py**  
Applies trained enhancement models to ExDark images.

**build_binary_classifier_dataset.py**  
Creates a dataset for raw vs enhanced classification.

**train_binary_classifier.py**  
Trains the binary classifier extension experiment.

**export_sample_pairs.py**  
Exports example image pairs used in the final presentation.

---

# Key Output Files

Enhancement runs produce:

```
metrics.csv
checkpoints/best.pt
plots/
sample_triplets/
```

Evaluation results are stored in:

```
runs/full_project/eval_best_rgb
```

Binary classifier outputs include:

```
runs/full_project/classifier_binary
 ├── metrics.csv
 ├── test_results.csv
 ├── loss_curve.png
 └── val_acc_curve.png
```

---

# Reproducibility

All experiments are reproducible using the provided scripts.

The included results (in a separate file containing slides) were generated in March 2026 using:

```
NVIDIA RTX 5070 Ti
PyTorch CUDA build
Mixed precision training
```

Due to GPU differences and random seeds, small variations in results may occur when reproducing the experiments.

---

# Summary

This project demonstrates a complete low-light image enhancement research pipeline including:

- multiple CNN architectures
- quantitative PSNR / SSIM evaluation
- qualitative visual analysis
- RGB vs grayscale comparison
- downstream classification experiment

The repository provides a reproducible framework for experimenting with deep learning methods for low-light image restoration.