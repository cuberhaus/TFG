# TFG

Bachelor's Thesis (Treball de Fi de Grau) code repository at FIB-UPC — deep learning system for polyp detection in colonoscopy videos using generative models for data augmentation.

## Overview

The project trains and evaluates object detection models (Faster R-CNN) on colonoscopy data, using generative models (CycleGAN, SPADE) to synthesize additional training images from the LDPolypVideo dataset. Hyperparameter optimization is performed with Optuna and Ray Tune.

## Structure

```
code/
├── src/
│   ├── train_and_save_model.py      # Main training script
│   ├── train_models.py              # Batch training
│   ├── optuna_train_model.py        # Optuna hyperparameter search
│   ├── raytune_train_model.py       # Ray Tune hyperparameter search
│   ├── evaluate_models.py           # Model evaluation
│   ├── test_model.py                # Testing script
│   ├── plot_losses.py               # Loss visualization
│   ├── cyclegan_train.py            # CycleGAN training
│   ├── cyclegan_test.py             # CycleGAN testing
│   ├── spade_train.py               # SPADE training
│   └── clases/
│       ├── cyclegan.py              # CycleGAN model definition
│       ├── model_utils.py           # Model utilities
│       ├── data_utils.py            # Data loading utilities
│       └── custom_dataset.py        # Custom PyTorch dataset
├── tmp/
│   ├── losses/                      # Training loss logs
│   └── tutorials/                   # PyTorch learning notebooks
└── requirements.txt                 # Python dependencies
docs/
└── papers/                          # Reference papers
```

## Tech Stack

- **Python** with PyTorch 2.1.2, torchvision 0.16.2
- **Optuna** and **Ray Tune** for hyperparameter optimization
- **OpenCV** for image processing
- **pycocotools** for COCO-format evaluation
