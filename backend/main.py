import os
import sys

# Fix for Intel MKL incompatible with libgomp error inside PyTorch subprocesses
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import io
import base64
import subprocess
import shutil
import json
import random
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional

# Add the 'code/src' directory to the path so we can import the original classes
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(BACKEND_DIR, "..")
SRC_DIR = os.path.join(PROJ_DIR, "code", "src")
sys.path.insert(0, SRC_DIR)

from clases.model_utils import get_model, load_model_with_hyperparams

# sys.executable may point to an Electron app (e.g. Cursor) when launched
# from an IDE terminal, so resolve the real Python interpreter.
def _find_python() -> str:
    if os.path.basename(sys.executable).startswith("python"):
        return sys.executable
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            return path
    return "python3"

PYTHON = _find_python()
import signal
import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image

app = FastAPI(title="TFG Polyp Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVED_MODELS_DIR = os.path.join(PROJ_DIR, "code", "out", "saved_models")
CSV_PATH = os.path.join(PROJ_DIR, "code", "out", "model_performances.csv")
LOSSES_DIR = os.path.join(PROJ_DIR, "code", "out", "losses")

class ModelInfo(BaseModel):
    filename: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    source_path: str

class DetectionBox(BaseModel):
    score: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class PredictionResponse(BaseModel):
    boxes: List[DetectionBox]
    image_base64: str

# --- Training State ---
training_state = {
    "is_training": False,
    "current_model": None,
    "message": "Idle",
    "pid": None,
}

class TrainingRequest(BaseModel):
    model_name: str
    batch_size: int
    lr: float
    weight_decay: float
    num_epochs: int
    max_samples: Optional[int] = None
    debug: bool = False

def run_training_script(req: TrainingRequest):
    global training_state
    
    # Pre-check for training dataset
    import glob
    train_ann_dir = os.path.join(SRC_DIR, "../data/TrainValid/Annotations")
    train_img_dir = os.path.join(SRC_DIR, "../data/TrainValid/Images")
    if not os.path.exists(train_ann_dir) or not glob.glob(os.path.join(train_ann_dir, "*", "*.*")):
        training_state["is_training"] = False
        training_state["message"] = "Error: Training dataset is empty. Please upload annotations to data/TrainValid/Annotations and images to data/TrainValid/Images."
        training_state["current_model"] = None
        return
        
    training_state["is_training"] = True
    training_state["current_model"] = req.model_name
    training_state["message"] = f"Training {req.model_name}..."
    
    script_path = os.path.join(SRC_DIR, "train_and_save_model.py")
    params = {
        "BATCH_SIZE": req.batch_size,
        "LR": req.lr,
        "WEIGHT_DECAY": req.weight_decay,
        "NUM_EPOCHS": req.num_epochs,
        "CONFIDENCE_THRESHOLD": 0.5
    }
    
    cmd = [PYTHON, script_path, req.model_name, json.dumps(params)]
    if req.max_samples:
        cmd.extend(["--max-samples", str(req.max_samples)])
    if req.debug:
        cmd.append("--debug")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        training_state["pid"] = process.pid
        
        import collections
        output_lines = collections.deque(maxlen=30)
        for line in process.stdout:
            output_lines.append(line.strip())
            training_state["message"] = '\n'.join(output_lines)
            
        process.wait()
        
        last_lines = '\n'.join(output_lines)
        if process.returncode == 0:
            training_state["message"] = f"Training completed successfully for {req.model_name}.\n\n{last_lines}"
        elif process.returncode < 0:
            training_state["message"] = f"Training cancelled for {req.model_name}."
        else:
            training_state["message"] = f"Training failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        training_state["message"] = f"Error during training: {str(e)}"
    finally:
        training_state["is_training"] = False
        training_state["current_model"] = None
        training_state["pid"] = None

# --- Evaluation State ---
evaluation_state = {
    "is_evaluating": False,
    "message": "Idle",
    "pid": None,
}

def run_evaluation_script():
    global evaluation_state
    
    # Pre-check for evaluation datasets (train and test are both used in prepare_dataset)
    import glob
    train_ann_dir = os.path.join(SRC_DIR, "../data/TrainValid/Annotations")
    test_ann_dir = os.path.join(SRC_DIR, "../data/Test/Annotations")
    if not os.path.exists(train_ann_dir) or not glob.glob(os.path.join(train_ann_dir, "*", "*.*")):
        evaluation_state["is_evaluating"] = False
        evaluation_state["message"] = "Error: Training dataset is empty. The evaluation script requires data in data/TrainValid/Annotations."
        return
    if not os.path.exists(test_ann_dir) or not glob.glob(os.path.join(test_ann_dir, "*", "*.*")):
        evaluation_state["is_evaluating"] = False
        evaluation_state["message"] = "Error: Test dataset is empty. Please upload data to data/Test/Annotations and data/Test/Images."
        return

    evaluation_state["is_evaluating"] = True
    evaluation_state["message"] = "Evaluating all models in the saved models directory..."
    
    script_path = os.path.join(SRC_DIR, "evaluate_models.py")
    
    try:
        process = subprocess.Popen(
            [PYTHON, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        evaluation_state["pid"] = process.pid
        
        import collections
        output_lines = collections.deque(maxlen=30)
        for line in process.stdout:
            output_lines.append(line.strip())
            evaluation_state["message"] = '\n'.join(output_lines)
            
        process.wait()
        
        last_lines = '\n'.join(output_lines)
        if process.returncode == 0:
            evaluation_state["message"] = f"Evaluation completed successfully.\n\n{last_lines}"
        elif process.returncode < 0:
            evaluation_state["message"] = "Evaluation cancelled."
        else:
            evaluation_state["message"] = f"Evaluation failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        evaluation_state["message"] = f"Error during evaluation: {str(e)}"
    finally:
        evaluation_state["is_evaluating"] = False
        evaluation_state["pid"] = None

# --- Dataset Preparation State ---
prep_state = {
    "is_running": False,
    "message": "Idle",
}

def _generate_single_mask(args):
    """Worker function for parallel mask generation."""
    img_path, ann_path, mask_out_path = args
    import cv2
    import numpy as np
    from clases.custom_dataset import parse_annotation

    img = cv2.imread(img_path)
    if img is None:
        return 0
    bboxes = parse_annotation(ann_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 255
    cv2.imwrite(mask_out_path, mask)
    return 1


def run_prepare_cyclegan():
    """Generate masks from annotations, then copy into CycleGAN folder structure."""
    global prep_state
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    import multiprocessing

    prep_state["is_running"] = True
    prep_state["message"] = "Starting dataset preparation..."

    data_root = os.path.join(PROJ_DIR, "code", "data")
    splits = [
        ("TrainValid", "train"),
        ("Test", "test"),
    ]
    num_workers = min(multiprocessing.cpu_count(), 16)

    try:
        # Step 1: Collect all mask jobs, then run in parallel with ProcessPoolExecutor
        mask_jobs = []
        for split_dir, _ in splits:
            root = os.path.join(data_root, split_dir)
            images_root = os.path.join(root, "Images")
            ann_root = os.path.join(root, "Annotations")
            masks_root = os.path.join(root, "Masks")

            if not os.path.isdir(images_root) or not os.path.isdir(ann_root):
                prep_state["message"] = f"Skipping {split_dir}: Images or Annotations folder missing."
                continue

            for video_id in sorted(os.listdir(images_root)):
                img_dir = os.path.join(images_root, video_id)
                ann_dir = os.path.join(ann_root, video_id)
                mask_dir = os.path.join(masks_root, video_id)
                if not os.path.isdir(img_dir):
                    continue

                os.makedirs(mask_dir, exist_ok=True)
                for fname in os.listdir(img_dir):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    ann_path = os.path.join(ann_dir, fname.replace(".jpg", ".txt"))
                    if not os.path.isfile(ann_path):
                        continue
                    mask_name = os.path.splitext(fname)[0] + "_mask.png"
                    mask_jobs.append((
                        os.path.join(img_dir, fname),
                        ann_path,
                        os.path.join(mask_dir, mask_name),
                    ))

        prep_state["message"] = f"Step 1/2 — Generating {len(mask_jobs):,} masks across {num_workers} processes..."

        total_masks = 0
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i, result in enumerate(pool.map(_generate_single_mask, mask_jobs, chunksize=256), 1):
                total_masks += result
                if i % 2000 == 0:
                    prep_state["message"] = f"Step 1/2 — Generated {total_masks:,}/{len(mask_jobs):,} masks..."

        prep_state["message"] = f"Step 1/2 done — {total_masks:,} masks generated. Copying to CycleGAN structure..."

        # Step 2: Parallel file copy with ThreadPoolExecutor (I/O bound)
        import shutil as sh
        target_names = ["PolypDataset", "PolypDatasetSPADE"]
        copy_jobs = []

        for target_name in target_names:
            target_root = os.path.join(data_root, target_name)
            for split_dir, mode in splits:
                root = os.path.join(data_root, split_dir)
                images_root = os.path.join(root, "Images")
                masks_root = os.path.join(root, "Masks")

                dir_a = os.path.join(target_root, f"{mode}A")
                dir_b = os.path.join(target_root, f"{mode}B")
                os.makedirs(dir_a, exist_ok=True)
                os.makedirs(dir_b, exist_ok=True)

                if not os.path.isdir(images_root) or not os.path.isdir(masks_root):
                    continue

                for video_id in sorted(os.listdir(masks_root)):
                    mask_dir = os.path.join(masks_root, video_id)
                    img_dir = os.path.join(images_root, video_id)
                    if not os.path.isdir(mask_dir):
                        continue

                    for mask_file in os.listdir(mask_dir):
                        if not mask_file.endswith(".png"):
                            continue
                        flat_name = mask_file.replace("_mask", "")
                        copy_jobs.append((os.path.join(mask_dir, mask_file), os.path.join(dir_a, f"{video_id}_{flat_name}")))

                    if os.path.isdir(img_dir):
                        for img_file in os.listdir(img_dir):
                            if not img_file.lower().endswith(".jpg"):
                                continue
                            copy_jobs.append((os.path.join(img_dir, img_file), os.path.join(dir_b, f"{video_id}_{img_file}")))

        prep_state["message"] = f"Step 2/2 — Copying {len(copy_jobs):,} files across {num_workers} threads..."

        total_copied = 0
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(sh.copy2, src, dst) for src, dst in copy_jobs]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                total_copied += 1
                if total_copied % 5000 == 0:
                    prep_state["message"] = f"Step 2/2 — Copied {total_copied:,}/{len(copy_jobs):,} files..."

        prep_state["message"] = (
            f"Done! Generated {total_masks:,} masks and copied {total_copied:,} files "
            f"into PolypDataset and PolypDatasetSPADE."
        )
    except Exception as e:
        import traceback
        prep_state["message"] = f"Error: {str(e)}\n{traceback.format_exc()}"
    finally:
        prep_state["is_running"] = False

@app.get("/api/prepare/status")
def get_prepare_status():
    return prep_state

@app.post("/api/prepare")
def start_prepare(background_tasks: BackgroundTasks):
    if prep_state["is_running"]:
        raise HTTPException(status_code=400, detail="Preparation is already running.")
    background_tasks.add_task(run_prepare_cyclegan)
    return {"message": "Dataset preparation started."}

# --- Generative Augmentation State ---
gen_state = {
    "is_running": False,
    "current_task": None,
    "message": "Idle",
    "pid": None,
}

class GenRequest(BaseModel):
    task_type: str
    experiment_name: Optional[str] = "mask2polyp"
    epoch: Optional[str] = "latest"
    # CycleGAN params
    batch_size: Optional[int] = 4
    n_epochs: Optional[int] = 5
    lr: Optional[float] = 0.0002
    netG: Optional[str] = "resnet_9blocks"
    load_size: Optional[int] = 286
    crop_size: Optional[int] = 256
    max_dataset_size: Optional[int] = None
    # SPADE params
    spade_batch_size: Optional[int] = 1
    spade_niter: Optional[int] = 50
    spade_niter_decay: Optional[int] = 0
    spade_lr: Optional[float] = 0.0002
    spade_netG: Optional[str] = "spade"
    spade_load_size: Optional[int] = 1024
    spade_crop_size: Optional[int] = 512
    spade_max_dataset_size: Optional[int] = None

def run_generative_script(req: GenRequest):
    global gen_state
    gen_state["is_running"] = True
    gen_state["current_task"] = req.task_type
    gen_state["message"] = f"Starting {req.task_type}..."
    
    script_name = ""
    cmd_args = []
    if req.task_type == "train_cyclegan":
        script_name = "cyclegan_train.py"
        cmd_args.extend([
            "--batch-size", str(req.batch_size),
            "--n-epochs", str(req.n_epochs),
            "--lr", str(req.lr),
            "--netG", req.netG,
            "--load-size", str(req.load_size),
            "--crop-size", str(req.crop_size),
        ])
        if req.max_dataset_size is not None:
            cmd_args.extend(["--max-dataset-size", str(req.max_dataset_size)])
    elif req.task_type == "test_cyclegan":
        script_name = "cyclegan_test.py"
        if req.experiment_name:
            cmd_args.extend(["--name", req.experiment_name])
        if req.epoch:
            cmd_args.extend(["--epoch", req.epoch])
    elif req.task_type == "train_spade":
        script_name = "spade_train.py"
        cmd_args.extend([
            "--batch-size", str(req.spade_batch_size),
            "--niter", str(req.spade_niter),
            "--niter-decay", str(req.spade_niter_decay),
            "--lr", str(req.spade_lr),
            "--netG", req.spade_netG,
            "--load-size", str(req.spade_load_size),
            "--crop-size", str(req.spade_crop_size),
        ])
        if req.spade_max_dataset_size is not None:
            cmd_args.extend(["--max-dataset-size", str(req.spade_max_dataset_size)])
    else:
        gen_state["message"] = f"Unknown task type: {req.task_type}"
        gen_state["is_running"] = False
        gen_state["current_task"] = None
        return
        
    script_path = os.path.join(SRC_DIR, script_name)
    
    # Pre-check dataset folders for empty images
    try:
        import glob
        if req.task_type == "train_cyclegan":
            data_dir = os.path.join(SRC_DIR, "../data/PolypDataset")
            trainA_files = glob.glob(os.path.join(data_dir, "trainA", "*.*"))
            trainB_files = glob.glob(os.path.join(data_dir, "trainB", "*.*"))
            if not trainA_files or not trainB_files:
                gen_state["message"] = "Error: Training dataset is empty. Please upload images to PolypDataset/trainA and trainB directories."
                gen_state["is_running"] = False
                gen_state["current_task"] = None
                return
        elif req.task_type == "test_cyclegan":
            data_dir = os.path.join(SRC_DIR, "../data/PolypDataset")
            testA_files = glob.glob(os.path.join(data_dir, "testA", "*.*"))
            if not testA_files:
                gen_state["message"] = "Error: Test dataset is empty. Please upload images to PolypDataset/testA directory."
                gen_state["is_running"] = False
                gen_state["current_task"] = None
                return
        elif req.task_type == "train_spade":
            data_dir = os.path.join(SRC_DIR, "../data/PolypDatasetSPADE")
            trainA_files = glob.glob(os.path.join(data_dir, "trainA", "*.*"))
            trainB_files = glob.glob(os.path.join(data_dir, "trainB", "*.*"))
            if not trainA_files or not trainB_files:
                gen_state["message"] = "Error: SPADE Training dataset is empty. Please upload images to PolypDatasetSPADE/trainA and trainB directories."
                gen_state["is_running"] = False
                gen_state["current_task"] = None
                return
    except Exception as e:
        print(f"Error checking directories: {e}")

    try:
        process = subprocess.Popen(
            [PYTHON, script_path] + cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        gen_state["pid"] = process.pid
        
        import collections
        output_lines = collections.deque(maxlen=20)
        for line in process.stdout:
            output_lines.append(line.strip())
            gen_state["message"] = '\n'.join(output_lines)
            
        process.wait()
        
        last_lines = '\n'.join(output_lines)
        if process.returncode == 0:
            gen_state["message"] = f"{req.task_type} completed successfully.\n\nFinal Output:\n{last_lines}"
        elif process.returncode < 0:
            gen_state["message"] = f"{req.task_type} cancelled."
        else:
            gen_state["message"] = f"{req.task_type} failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        gen_state["message"] = f"Error during {req.task_type}: {str(e)}"
    finally:
        gen_state["is_running"] = False
        gen_state["current_task"] = None
        gen_state["pid"] = None

@app.get("/api/generate/cyclegan-experiments")
def get_cyclegan_experiments():
    cyclegan_dir = os.path.join(PROJ_DIR, "code", "tmp", "pytorch-CycleGAN-and-pix2pix", "checkpoints")
    if not os.path.isdir(cyclegan_dir):
        return {"experiments": {}}
    
    experiments = {}
    for exp_folder in os.listdir(cyclegan_dir):
        exp_path = os.path.join(cyclegan_dir, exp_folder)
        if os.path.isdir(exp_path):
            epochs = set()
            for file in os.listdir(exp_path):
                if file.endswith("_net_G_A.pth") or file.endswith("_net_G.pth"):
                    # Extract the prefix before '_net_'
                    epoch_str = file.split("_net_")[0]
                    epochs.add(epoch_str)
            # Sort the epochs, putting 'latest' at the top, then sorting the rest numerically
            epochs_list = list(epochs)
            epochs_list.sort(key=lambda x: -1 if x == 'latest' else (int(x) if x.isdigit() else 9999))
            
            # If a folder has no valid .pth files, we can still list it, but give it an empty list
            if not epochs_list:
                 epochs_list = ["latest"] # Fallback
            experiments[exp_folder] = epochs_list
            
    return {"experiments": experiments}

@app.get("/api/generate/results")
def get_generation_results(experiment: str = None, epoch: str = "latest"):
    results_dir = os.path.join(PROJ_DIR, "code", "tmp", "pytorch-CycleGAN-and-pix2pix", "results")
    if not os.path.isdir(results_dir):
        return {"images": []}
    
    images_data = []
    
    # If no experiment specified, grab the first one available
    if not experiment:
        experiments = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
        if not experiments:
            return {"images": []}
        experiment = experiments[0]
        
    exp_dir = os.path.join(results_dir, experiment)
    if not os.path.isdir(exp_dir):
        return {"images": []}
        
    test_dir = os.path.join(exp_dir, f"test_{epoch}")
    if not os.path.isdir(test_dir):
        # try to find any test_ folder
        test_folders = [f for f in os.listdir(exp_dir) if f.startswith("test_") and os.path.isdir(os.path.join(exp_dir, f))]
        if not test_folders:
            return {"images": []}
        test_dir = os.path.join(exp_dir, test_folders[-1])
        
    images_folder = os.path.join(test_dir, "images")
    if not os.path.isdir(images_folder):
        return {"images": []}
        
    # Group images by base name (without _real or _fake suffix)
    image_groups = {}
    for img_name in os.listdir(images_folder):
        if img_name.endswith(".png") or img_name.endswith(".jpg"):
            if "_real" in img_name:
                base_name = img_name.replace("_real", "")
                group_key = "real"
            elif "_fake" in img_name:
                base_name = img_name.replace("_fake", "")
                group_key = "fake"
            else:
                base_name = img_name
                group_key = "other"
                
            if base_name not in image_groups:
                image_groups[base_name] = {}
                
            img_path = os.path.join(images_folder, img_name)
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                image_groups[base_name][group_key] = f"data:image/png;base64,{encoded_string}"
                
    for base_name, group in image_groups.items():
        images_data.append({
            "base_name": base_name,
            "real": group.get("real"),
            "fake": group.get("fake")
        })
        
    return {"images": images_data, "experiment": experiment, "test_dir": os.path.basename(test_dir)}

@app.get("/api/generate/status")
def get_generate_status():
    return gen_state

@app.post("/api/generate")
def start_generation(req: GenRequest, background_tasks: BackgroundTasks):
    if gen_state["is_running"]:
        raise HTTPException(status_code=400, detail=f"Task {gen_state['current_task']} is already running.")
    background_tasks.add_task(run_generative_script, req)
    return {"message": "Generation task started."}

@app.post("/api/generate/cancel")
def cancel_generation():
    pid = gen_state.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="No generative task is running.")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"message": "Generation cancellation requested."}

# --- Hyperparameter Tuning State ---
hpo_state = {
    "is_tuning": False,
    "current_model": None,
    "message": "Idle",
    "pid": None,
}

class HPOResquest(BaseModel):
    model_name: str
    num_trials: int
    max_epochs: Optional[int] = 5
    max_samples: Optional[int] = None
    debug: bool = False

def run_hpo_script(req: HPOResquest):
    global hpo_state
    
    # Pre-check for HPO dataset
    import glob
    train_ann_dir = os.path.join(SRC_DIR, "../data/TrainValid/Annotations")
    if not os.path.exists(train_ann_dir) or not glob.glob(os.path.join(train_ann_dir, "*", "*.*")):
        hpo_state["is_tuning"] = False
        hpo_state["message"] = "Error: Training dataset is empty. Please upload annotations to data/TrainValid/Annotations and images to data/TrainValid/Images."
        hpo_state["current_model"] = None
        return

    hpo_state["is_tuning"] = True
    hpo_state["current_model"] = req.model_name
    hpo_state["message"] = f"Starting Hyperparameter Tuning for {req.model_name} with {req.num_trials} trials...\n"
    
    script_path = os.path.join(SRC_DIR, "optuna_train_model.py")
    
    try:
        process = subprocess.Popen(
            [PYTHON, script_path, req.model_name,
             "--n-trials", str(req.num_trials),
             "--max-epochs", str(req.max_epochs),
             *(["--max-samples", str(req.max_samples)] if req.max_samples else []),
             *(["--debug"] if req.debug else []),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        hpo_state["pid"] = process.pid
        
        import collections
        output_lines = collections.deque(maxlen=20)
        for line in process.stdout:
            output_lines.append(line.strip())
            hpo_state["message"] = '\n'.join(output_lines)
            
        process.wait()
        
        last_lines = '\n'.join(output_lines)
        if process.returncode == 0:
            hpo_state["message"] = f"Tuning completed successfully for {req.model_name}.\nCheck best_hyperparameters.csv in the codebase.\n\nFinal Output:\n{last_lines}"
        elif process.returncode < 0:
            hpo_state["message"] = f"Tuning cancelled for {req.model_name}."
        else:
            hpo_state["message"] = f"Tuning failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        hpo_state["message"] = f"Error during tuning: {str(e)}"
    finally:
        hpo_state["is_tuning"] = False
        hpo_state["current_model"] = None
        hpo_state["pid"] = None

@app.get("/api/hpo/status")
def get_hpo_status():
    return hpo_state

@app.post("/api/hpo/start")
def start_hpo(req: HPOResquest, background_tasks: BackgroundTasks):
    if hpo_state["is_tuning"]:
        raise HTTPException(status_code=400, detail=f"A tuning job is already running for {hpo_state['current_model']}.")
    background_tasks.add_task(run_hpo_script, req)
    return {"message": "Hyperparameter tuning started."}

@app.post("/api/hpo/cancel")
def cancel_hpo():
    pid = hpo_state.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="No HPO process is running.")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"message": "HPO cancellation requested."}

# --- Dataset Explorer Endpoint ---
@app.get("/api/dataset/{split}")
def get_dataset_images(split: str, page: int = 1, limit: int = 12):
    if split == "train":
        root_dir = os.path.join(SRC_DIR, "../data/TrainValid")
    elif split == "test":
        root_dir = os.path.join(SRC_DIR, "../data/Test")
    else:
        raise HTTPException(status_code=400, detail="Invalid split. Use 'train' or 'test'.")
        
    ann_root = os.path.join(root_dir, "Annotations")
    img_root = os.path.join(root_dir, "Images")
    
    if not os.path.isdir(ann_root) or not os.path.isdir(img_root):
        return {"images": [], "total": 0, "page": page, "limit": limit}
        
    # Collect all image/annotation pairs
    all_items = []
    
    try:
        for subdir in sorted(os.listdir(ann_root)):
            ann_sub = os.path.join(ann_root, subdir)
            img_sub = os.path.join(img_root, subdir)
            
            if os.path.isdir(ann_sub) and os.path.isdir(img_sub):
                for filename in sorted(os.listdir(ann_sub)):
                    if filename.endswith(".txt"):
                        img_filename = filename.replace(".txt", ".jpg")
                        img_path = os.path.join(img_sub, img_filename)
                        ann_path = os.path.join(ann_sub, filename)
                        
                        if os.path.exists(img_path):
                            all_items.append({
                                "id": f"{subdir}/{img_filename}",
                                "img_path": img_path,
                                "ann_path": ann_path,
                                "subdir": subdir,
                                "filename": img_filename
                            })
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return {"images": [], "total": 0, "page": page, "limit": limit}
        
    total_items = len(all_items)
    
    # Pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = all_items[start_idx:end_idx]
    
    results = []
    for item in paginated_items:
        # Parse annotation
        boxes = []
        try:
            with open(item["ann_path"], 'r') as f:
                lines = f.readlines()
            if lines:
                num_objs = int(lines[0].strip())
                for i in range(1, num_objs + 1):
                    if i < len(lines):
                        vals = list(map(int, lines[i].strip().split()))
                        if len(vals) == 4:
                            boxes.append(vals) # [xmin, ymin, xmax, ymax]
        except Exception as e:
            print(f"Error parsing {item['ann_path']}: {e}")
            
        # Read and encode image
        try:
            with Image.open(item["img_path"]) as img:
                width, height = img.size
                # Convert to RGB if not
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for preview to save bandwidth (e.g., max 500px width)
                img.thumbnail((500, 500))
                
                # Calculate scale factor to adjust boxes
                scale_x = img.width / width
                scale_y = img.height / height
                
                scaled_boxes = []
                for b in boxes:
                    scaled_boxes.append([
                        int(b[0] * scale_x),
                        int(b[1] * scale_y),
                        int(b[2] * scale_x),
                        int(b[3] * scale_y)
                    ])
                    
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=80)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                results.append({
                    "id": item["id"],
                    "subdir": item["subdir"],
                    "filename": item["filename"],
                    "image": f"data:image/jpeg;base64,{img_str}",
                    "width": img.width,
                    "height": img.height,
                    "boxes": scaled_boxes,
                    "original_boxes": boxes,
                    "original_width": width,
                    "original_height": height
                })
        except Exception as e:
            print(f"Error processing image {item['img_path']}: {e}")
            
    return {
        "images": results,
        "total": total_items,
        "page": page,
        "limit": limit,
        "total_pages": (total_items + limit - 1) // limit
    }

ALLOWED_DATA_PREFIXES = ['TrainValid', 'Test', 'PolypDataset', 'PolypDatasetSPADE']
DATA_DIR = os.path.normpath(os.path.join(PROJ_DIR, "code", "data"))

@app.post("/api/dataset/upload")
async def upload_dataset_files(
    files: List[UploadFile] = File(...),
    relative_paths: str = Form(...)
):
    paths = json.loads(relative_paths)
    if len(files) != len(paths):
        raise HTTPException(status_code=400, detail="File count doesn't match path count")

    saved = 0
    skipped = 0

    for upload_file, rel_path in zip(files, paths):
        top_folder = rel_path.split('/')[0]
        if top_folder not in ALLOWED_DATA_PREFIXES:
            skipped += 1
            continue

        target = os.path.normpath(os.path.join(DATA_DIR, rel_path))
        if not target.startswith(DATA_DIR + os.sep):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(target), exist_ok=True)
        content = await upload_file.read()
        with open(target, 'wb') as f:
            f.write(content)
        saved += 1

    return {"saved": saved, "skipped": skipped, "total": len(files)}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/losses/files")
def get_loss_files():
    if not os.path.isdir(LOSSES_DIR):
        return {"files": [], "source_path": LOSSES_DIR}
    
    files = [f for f in os.listdir(LOSSES_DIR) if f.endswith("_losses.txt")]
    return {"files": files, "source_path": LOSSES_DIR}

class LossDataRequest(BaseModel):
    files: List[str]

@app.post("/api/losses/data")
def get_loss_data(req: LossDataRequest):
    data = []
    MAX_POINTS = 500  # Downsample to a maximum number of points to prevent frontend lag
    
    for filename in req.files:
        filepath = os.path.join(LOSSES_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                values = [float(line.strip()) for line in f if line.strip()]
            
            # Downsample if there are too many points (crucial for batch_losses.txt)
            if len(values) > MAX_POINTS:
                step = len(values) / MAX_POINTS
                # Take evenly spaced samples, but ensure we keep the first and last points
                sampled_indices = [int(i * step) for i in range(MAX_POINTS)]
                if sampled_indices[-1] != len(values) - 1:
                    sampled_indices[-1] = len(values) - 1
                
                # To make the graph smoother, we can average the points around the sample instead of just picking one
                # A simple moving average downsample
                downsampled_values = []
                window_size = max(1, int(step))
                for i in sampled_indices:
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(values), i + window_size // 2 + 1)
                    window = values[start_idx:end_idx]
                    avg_val = sum(window) / len(window) if window else values[i]
                    downsampled_values.append(avg_val)
                    
                values = downsampled_values

            short_name = filename.replace("_epoch_losses.txt", "").replace("_batch_losses.txt", "")
            data.append({
                "filename": filename,
                "short_name": short_name,
                "values": values
            })
    return {"data": data}

@app.get("/api/evaluate/status")
def get_evaluation_status():
    return evaluation_state

@app.post("/api/evaluate")
def start_evaluation(background_tasks: BackgroundTasks):
    if evaluation_state["is_evaluating"]:
        raise HTTPException(status_code=400, detail="Model evaluation is already in progress.")
    background_tasks.add_task(run_evaluation_script)
    return {"message": "Evaluation started."}

@app.post("/api/evaluate/cancel")
def cancel_evaluation():
    pid = evaluation_state.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="No evaluation process is running.")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"message": "Evaluation cancellation requested."}

@app.get("/api/models", response_model=ModelsResponse)
def get_models():
    if not os.path.isdir(SAVED_MODELS_DIR):
        return {"models": [], "source_path": SAVED_MODELS_DIR}
    models = [f for f in os.listdir(SAVED_MODELS_DIR) if not f.startswith(".")]
    return {"models": [{"filename": m} for m in models], "source_path": SAVED_MODELS_DIR}

@app.get("/api/train/status")
def get_training_status():
    return training_state

@app.post("/api/train")
def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="A model is already being trained.")
    background_tasks.add_task(run_training_script, req)
    return {"message": "Training started."}

@app.post("/api/train/cancel")
def cancel_training():
    pid = training_state.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="No training process is running.")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"message": "Training cancellation requested."}

@app.get("/api/performance")
def get_performance():
    if not os.path.exists(CSV_PATH):
        return {"data": [], "source_path": CSV_PATH}
    
    df = pd.read_csv(CSV_PATH)
    
    def _f1(row):
        ap, ar = row.get("AP_50_95_all", 0), row.get("AR_50_95_all_maxDets_100", 0)
        return 2 * ap * ar / (ap + ar) if (ap + ar) > 0 else 0.0
        
    df["F1"] = df.apply(_f1, axis=1)
    df["LR_fmt"] = df["LR"].map(lambda x: f"{x:.2e}")
    df["Config"] = (
        df["Model"]
        + " bs="
        + df["BATCH_SIZE"].astype(str)
        + " lr="
        + df["LR_fmt"]
        + " ep="
        + df["NUM_EPOCHS"].astype(str)
    )
    
    return {"data": df.to_dict(orient="records"), "source_path": CSV_PATH}

_model_cache = {}

def _get_cached_model(model_arch: str, model_file: str):
    """Load model once and cache it on GPU for fast repeated inference."""
    cache_key = f"{model_arch}:{model_file}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    num_classes = 2
    model = get_model(model_arch, num_classes)
    model, _, _ = load_model_with_hyperparams(model, model_file, load_dir=SAVED_MODELS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if len(_model_cache) >= 3:
        oldest = next(iter(_model_cache))
        del _model_cache[oldest]
        torch.cuda.empty_cache()

    _model_cache[cache_key] = (model, device)
    return model, device

_inference_transform = transforms.Compose([
    transforms.Resize((560, 480)),
    transforms.ToTensor()
])

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_arch: str = Form("FasterRCNN"),
    model_file: str = Form(...),
    confidence: float = Form(0.5)
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    model, device = _get_cached_model(model_arch, model_file)
    use_amp = device.type == 'cuda'
    img_tensor = _inference_transform(img).unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        output = model(img_tensor)[0]

    keep = output["scores"] >= confidence
    boxes = output["boxes"][keep].cpu()
    scores = output["scores"][keep].cpu()

    result_boxes = []
    if len(boxes) > 0:
        for i in range(len(boxes)):
            result_boxes.append(DetectionBox(
                score=scores[i].item(),
                x_min=boxes[i][0].item(),
                y_min=boxes[i][1].item(),
                x_max=boxes[i][2].item(),
                y_max=boxes[i][3].item()
            ))

        img_byte = _inference_transform(img).mul(255).byte()
        label_strings = [f"polyp {s:.2f}" for s in scores.tolist()]
        drawn = draw_bounding_boxes(img_byte, boxes, labels=label_strings, colors="red", width=2)
        result_img = to_pil_image(drawn)
    else:
        result_img = img.resize((480, 560))

    buffered = io.BytesIO()
    result_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return PredictionResponse(boxes=result_boxes, image_base64=img_str)

@app.post("/api/predict/batch")
async def predict_batch(
    model_arch: str = Form("FasterRCNN"),
    model_file: str = Form(...),
    confidence: float = Form(0.5)
):
    test_img_dir = os.path.join(SRC_DIR, "../data/Test/Images")
    if not os.path.exists(test_img_dir):
        raise HTTPException(status_code=400, detail="Test dataset not found.")

    all_images = []
    for subdir in os.listdir(test_img_dir):
        subdir_path = os.path.join(test_img_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(subdir_path, file))

    if not all_images:
        raise HTTPException(status_code=400, detail="No images found in Test dataset.")

    selected_images = random.sample(all_images, min(9, len(all_images)))

    model, device = _get_cached_model(model_arch, model_file)
    use_amp = device.type == 'cuda'

    pil_images = []
    tensors = []
    for img_path in selected_images:
        try:
            img = Image.open(img_path).convert("RGB")
            pil_images.append((img_path, img))
            tensors.append(_inference_transform(img))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    if not tensors:
        return {"results": []}

    batch_tensor = torch.stack(tensors).to(device, non_blocking=True)

    results = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(list(batch_tensor))

    for (img_path, img), output in zip(pil_images, outputs):
        keep = output["scores"] >= confidence
        boxes = output["boxes"][keep].cpu()
        scores = output["scores"][keep].cpu()

        result_boxes = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                result_boxes.append(DetectionBox(
                    score=scores[i].item(),
                    x_min=boxes[i][0].item(),
                    y_min=boxes[i][1].item(),
                    x_max=boxes[i][2].item(),
                    y_max=boxes[i][3].item()
                ))

            img_byte = _inference_transform(img).mul(255).byte()
            label_strings = [f"polyp {s:.2f}" for s in scores.tolist()]
            drawn = draw_bounding_boxes(img_byte, boxes, labels=label_strings, colors="red", width=2)
            result_img = to_pil_image(drawn)
        else:
            result_img = img.resize((480, 560))

        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        results.append({
            "filename": os.path.basename(img_path),
            "boxes": result_boxes,
            "image_base64": img_str
        })

    return {"results": results}
