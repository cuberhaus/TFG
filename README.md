# TFG — Polyp Detection with Generative Data Augmentation

Bachelor's Thesis (*Treball de Fi de Grau*) at [FIB-UPC](https://www.fib.upc.edu/) — a deep learning system for polyp detection in colonoscopy videos, using generative models for synthetic data augmentation.

## Abstract

This project investigates generative deep learning models for image-to-image translation to improve polyp detection in colonoscopies. Medical imaging datasets often suffer from scarcity, limited diversity, and variable quality. By leveraging generative models (CycleGAN, SPADE) to synthesize realistic training images, the project augments the existing LDPolypVideo dataset and trains object detection models (Faster R-CNN, RetinaNet, SSD) with improved precision and recall. Hyperparameter optimization is performed with Optuna and Ray Tune.

**Keywords:** Machine Learning, Computer Vision, Generative Models, Image-to-Image Translation, Object Detection

## Prerequisites

| Requirement | Version |
| --- | --- |
| Python | 3.11+ |
| CUDA (optional, for GPU training) | 11.8+ |
| conda or pip | latest |

A CUDA-capable GPU is strongly recommended for training. CPU-only inference is supported but slow.

## Installation

```bash
git clone https://github.com/cuberhaus/TFG.git
cd TFG

# Install everything (backend pip + frontend npm)
make install

# Or manually:
cd backend && pip install -r requirements.txt
cd ../frontend && npm install
```

## Tests

Run the fast backend endpoint suite without models, datasets, or frontend dependencies:

```bash
make test-backend
```

CI installs the pinned test-only dependency contract in `backend/requirements-ci.txt`; the application dependency files remain `backend/requirements.txt` and `code/requirements.txt`.

## Dataset Setup

The project uses the [LDPolypVideo](https://github.com/dashishi/LDPolypVideo-Benchmark) dataset. Download and place it under `code/data/`:

```text
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

## Web Dashboard (recommended)

The project ships a full-stack web dashboard — a React frontend talking to a FastAPI backend — that wraps every feature below in a visual UI.

```bash
# Start both backend and frontend with one command:
make run

# Or start them separately:
make backend   # FastAPI on http://localhost:8082
make frontend  # Vite dev server on http://localhost:5173
```

The dashboard provides:

| Tab | Description |
| --- | --- |
| **Dataset Explorer** | Browse & upload datasets with drag-and-drop |
| **Model Training** | Train detection models with configurable parameters |
| **Model Evaluation** | Evaluate saved models (COCO metrics) |
| **Performance Explorer** | Compare AP, AR, F1 across all configurations |
| **Inference** | Single & batch image inference with bounding-box visualization |
| **Training Losses** | Plot epoch/batch losses for any training run |
| **Generative Augmentation** | Train/test CycleGAN and SPADE, browse generated images |
| **Hyperparameter Tuning** | Optuna HPO with live logs and result visualization |

## CLI Quick Start

All CLI commands are run from `code/`.

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

## Streamlit Dashboard (legacy)

A legacy Streamlit app is also available for quick exploration:

```bash
cd code
streamlit run src/app.py
```

## Project Structure

```text
TFG/
├── Makefile                              # run/install/clean targets for full stack
├── frontend/                             # React 19 + Vite + Tailwind UI
│   ├── src/
│   │   ├── api.ts                       # Shared axios instance (VITE_API_URL)
│   │   ├── App.tsx                      # Tab-based dashboard layout
│   │   └── components/                  # One component per dashboard tab
│   └── package.json
├── backend/                              # FastAPI REST API
│   ├── main.py                          # All endpoints, subprocess management
│   ├── run.sh                           # Uvicorn start script
│   └── requirements.txt
├── code/                                 # ML logic & standalone scripts
│   ├── src/
│   │   ├── app.py                       # Legacy Streamlit dashboard
│   │   ├── train_and_save_model.py      # Train a single model with CLI args
│   │   ├── train_models.py              # Batch training from JSON config
│   │   ├── optuna_train_model.py        # Optuna hyperparameter search
│   │   ├── raytune_train_model.py       # Ray Tune hyperparameter search
│   │   ├── evaluate_models.py           # Evaluate all saved models (COCO metrics)
│   │   ├── test_model.py               # Run inference and save detection images
│   │   ├── cyclegan_train.py            # Train CycleGAN for mask→polyp translation
│   │   ├── spade_train.py              # Train SPADE for mask→polyp translation
│   │   └── clases/
│   │       ├── model_utils.py           # Model loading, training loop, COCO eval
│   │       ├── custom_dataset.py        # PyTorch Dataset for LDPolypVideo
│   │       ├── cyclegan.py              # CycleGAN/SPADE setup & preparation
│   │       └── utils.py                 # Platform detection & helpers
│   ├── data/                            # Dataset root (not tracked in git)
│   ├── out/                             # Outputs: saved models, losses, predictions
│   └── requirements.txt                 # Python dependencies (ML-specific)
└── README.md
```

## Key Scripts Reference

| Script | Purpose | Key Arguments |
| --- | --- | --- |
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
| --- | --- | --- |
| **Faster R-CNN** | ResNet-50 FPN | Primary detector; best overall results |
| **RetinaNet** | ResNet-50 FPN v2 | Single-stage anchor-based alternative |
| **SSD Lite** | MobileNet V3 Large | Lightweight / mobile-friendly |

## Generative Models

| Model | Task |
| --- | --- |
| **CycleGAN** | Unpaired mask ↔ polyp image translation |
| **SPADE** | Spatially-adaptive mask → polyp synthesis |

## Tech Stack

| Layer | Technologies |
| --- | --- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, Recharts, Lucide React |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **ML** | PyTorch + torchvision, pycocotools (COCO metrics), OpenCV, Pillow |
| **HPO** | Optuna, Ray Tune |
| **Generative** | CycleGAN, SPADE (via bundled repos in `code/tmp/`) |

## License

This project was developed as a Bachelor's Thesis at FIB-UPC (Universitat Politecnica de Catalunya).
