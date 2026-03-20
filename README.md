# TFG — Polyp Detection with Generative Data Augmentation

Bachelor's Thesis (*Treball de Fi de Grau*) at [FIB-UPC](https://www.fib.upc.edu/) — a deep learning system for polyp detection in colonoscopy videos, using generative models for synthetic data augmentation.

## Abstract

This project investigates generative deep learning models for image-to-image translation to improve polyp detection in colonoscopies. Medical imaging datasets often suffer from scarcity, limited diversity, and variable quality. By leveraging generative models (CycleGAN, SPADE) to synthesize realistic training images, the project augments the existing LDPolypVideo dataset and trains object detection models (Faster R-CNN, RetinaNet, SSD) with improved precision and recall. Hyperparameter optimization is performed with Optuna and Ray Tune.

**Keywords:** Machine Learning, Computer Vision, Generative Models, Image-to-Image Translation, Object Detection

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| CUDA (optional, for GPU training) | 11.8+ |
| conda or pip | latest |

A CUDA-capable GPU is strongly recommended for training. CPU-only inference is supported but slow.

## Installation

```bash
git clone https://github.com/cuberhaus/TFG.git
cd TFG/code

# Option A: pip + venv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate tfg
```

## Dataset Setup

The project uses the [LDPolypVideo](https://github.com/dashishi/LDPolypVideo-Benchmark) dataset. Download and place it under `code/data/`:

```
code/data/
├── TrainValid/
│   └── TrainValid/
│       ├── Annotations/    # .txt bounding-box files
│       └── Images/         # .jpg colonoscopy frames
├── Test/
│   └── Test/
│       ├── Annotations/
│       └── Images/
└── PolypDataset/           # CycleGAN-format paired data
```

Each annotation file lists the number of polyps on line 1, followed by one bounding box per line (`x_min y_min x_max y_max`).

## Quick Start

All commands are run from `code/`.

### Train a single model

```bash
python src/train_and_save_model.py FasterRCNN \
  '{"BATCH_SIZE": 4, "LR": 0.005, "WEIGHT_DECAY": 0.0005, "NUM_EPOCHS": 10}' \
  --debug
```

### Batch training from a config file

```bash
python src/train_models.py configurations/model.json --debug
```

### Hyperparameter search with Optuna

```bash
python src/optuna_train_model.py --debug

# Background run for a specific architecture:
nohup python src/optuna_train_model.py FasterRCNN > optuna_train_model.log &
```

### Hyperparameter search with Ray Tune

```bash
python src/raytune_train_model.py --debug
```

### Evaluate all saved models

```bash
python src/evaluate_models.py
```

Results are written to `out/model_performances.csv`.

### Test a model (save detection images)

```bash
python src/test_model.py
```

### Train CycleGAN / SPADE generative models

```bash
python src/cyclegan_train.py
python src/cyclegan_test.py
python src/spade_train.py
```

### Data preparation helpers

```bash
python src/python_scripts/create_masks.py                       # Bounding boxes → binary masks
python src/python_scripts/copy_files_to_cyclegan_structure.py   # Reorganize for CycleGAN
```

## Streamlit Dashboard

An interactive Streamlit app lets you explore model results, run inference, and visualize training losses without writing code.

```bash
cd code
streamlit run src/app.py
```

The dashboard has three tabs:

| Tab | Description |
|---|---|
| **Performance Explorer** | Interactive charts comparing AP, AR, and F1 across all trained model configurations. Filter by batch size, learning rate, and epochs. |
| **Inference** | Upload a colonoscopy image, pick a saved model, adjust the confidence threshold, and see bounding-box detections. |
| **Training Losses** | Plot epoch and batch losses from `out/losses/` for any saved training run. |

## Project Structure

```
TFG/
├── code/
│   ├── src/
│   │   ├── app.py                        # Streamlit dashboard
│   │   ├── train_and_save_model.py       # Train a single model with CLI args
│   │   ├── train_models.py               # Batch training from JSON config
│   │   ├── optuna_train_model.py         # Optuna hyperparameter search
│   │   ├── raytune_train_model.py        # Ray Tune hyperparameter search
│   │   ├── evaluate_models.py            # Evaluate all saved models (COCO metrics)
│   │   ├── test_model.py                 # Run inference and save detection images
│   │   ├── plot_losses.py                # Plot training losses
│   │   ├── plot_evaluated_models.py      # Plot model comparison charts
│   │   ├── cyclegan_train.py             # Train CycleGAN for mask→polyp translation
│   │   ├── cyclegan_test.py              # Test CycleGAN outputs
│   │   ├── spade_train.py                # Train SPADE for mask→polyp translation
│   │   ├── clases/
│   │   │   ├── cyclegan.py               # CycleGAN/SPADE setup & preparation
│   │   │   ├── model_utils.py            # Model loading, training loop, COCO eval
│   │   │   ├── data_utils.py             # Data loading utilities
│   │   │   ├── custom_dataset.py         # PyTorch Dataset for LDPolypVideo
│   │   │   └── utils.py                  # Platform detection & helpers
│   │   ├── csv/                          # Result tables & formatting scripts
│   │   ├── j_notebooks/                  # Interactive Jupyter analysis notebooks
│   │   ├── python_scripts/               # One-off data preparation scripts
│   │   └── shell_scripts/                # Automation & remote sync scripts
│   ├── data/                             # Dataset root (not tracked in git)
│   ├── out/                              # Outputs: saved models, losses, predictions
│   ├── tmp/                              # Temp files & tutorial notebooks
│   ├── requirements.txt                  # Python dependencies
│   ├── Pipfile                           # Alternative Pipenv config
│   └── Makefile                          # Zip packaging
└── README.md                             # This file
```

## Key Scripts Reference

| Script | Purpose | Key Arguments |
|---|---|---|
| `train_and_save_model.py` | Train one model | `MODEL_NAME PARAMS_JSON [--debug]` |
| `train_models.py` | Batch training | `CONFIG_JSON [--debug]` |
| `optuna_train_model.py` | Optuna HPO | `[MODEL_NAME] [--debug]` |
| `raytune_train_model.py` | Ray Tune HPO | `[--debug]` |
| `evaluate_models.py` | COCO-format eval | Reads from `out/saved_models/` |
| `test_model.py` | Visual inference test | Saves to `out/testing_model/` |
| `cyclegan_train.py` | CycleGAN training | Batch size, epochs in script |
| `cyclegan_test.py` | CycleGAN generation | — |
| `spade_train.py` | SPADE training | — |
| `app.py` | Streamlit dashboard | `streamlit run src/app.py` |

## Detection Models

| Model | Backbone | Notes |
|---|---|---|
| **Faster R-CNN** | ResNet-50 FPN | Primary detector; best overall results |
| **RetinaNet** | ResNet-50 FPN v2 | Single-stage anchor-based alternative |
| **SSD Lite** | MobileNet V3 Large | Lightweight / mobile-friendly |

## Generative Models

| Model | Task |
|---|---|
| **CycleGAN** | Unpaired mask ↔ polyp image translation |
| **SPADE** | Spatially-adaptive mask → polyp synthesis |

## Tech Stack

- **PyTorch** 2.1.2 + **torchvision** 0.16.2
- **Optuna** and **Ray Tune** for hyperparameter optimization
- **OpenCV** for image processing
- **pycocotools** for COCO-format evaluation (AP, AR metrics)
- **Streamlit** for the interactive dashboard
- **Plotly** for interactive charts
- **Pillow** for image I/O

## License

This project was developed as a Bachelor's Thesis at FIB-UPC (Universitat Politecnica de Catalunya).
