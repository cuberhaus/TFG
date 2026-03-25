import os
import sys
import io
import base64
import subprocess
import json
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

SAVED_MODELS_DIR = os.path.join(PROJ_DIR, "out", "saved_models")
CSV_PATH = os.path.join(SRC_DIR, "csv", "model_performances.csv")
LOSSES_DIR = os.path.join(PROJ_DIR, "out", "losses")

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
}

class TrainingRequest(BaseModel):
    model_name: str
    batch_size: int
    lr: float
    weight_decay: float
    num_epochs: int

def run_training_script(req: TrainingRequest):
    global training_state
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
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path, req.model_name, json.dumps(params)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        import collections
        output_lines = collections.deque(maxlen=15)
        for line in process.stdout:
            output_lines.append(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            training_state["message"] = f"Training completed successfully for {req.model_name}."
        else:
            last_lines = '\n'.join(output_lines)
            training_state["message"] = f"Training failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        training_state["message"] = f"Error during training: {str(e)}"
    finally:
        training_state["is_training"] = False
        training_state["current_model"] = None

# --- Evaluation State ---
evaluation_state = {
    "is_evaluating": False,
    "message": "Idle",
}

def run_evaluation_script():
    global evaluation_state
    evaluation_state["is_evaluating"] = True
    evaluation_state["message"] = "Evaluating all models in the saved models directory..."
    
    script_path = os.path.join(SRC_DIR, "evaluate_models.py")
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        import collections
        output_lines = collections.deque(maxlen=15)
        for line in process.stdout:
            output_lines.append(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            evaluation_state["message"] = "Evaluation completed successfully. The Performance Explorer tab should now reflect the updated data."
        else:
            last_lines = '\n'.join(output_lines)
            evaluation_state["message"] = f"Evaluation failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        evaluation_state["message"] = f"Error during evaluation: {str(e)}"
    finally:
        evaluation_state["is_evaluating"] = False

# --- Generative Augmentation State ---
gen_state = {
    "is_running": False,
    "current_task": None,
    "message": "Idle",
}

class GenRequest(BaseModel):
    task_type: str

def run_generative_script(req: GenRequest):
    global gen_state
    gen_state["is_running"] = True
    gen_state["current_task"] = req.task_type
    gen_state["message"] = f"Starting {req.task_type}..."
    
    script_name = ""
    if req.task_type == "train_cyclegan":
        script_name = "cyclegan_train.py"
    elif req.task_type == "test_cyclegan":
        script_name = "cyclegan_test.py"
    elif req.task_type == "train_spade":
        script_name = "spade_train.py"
    else:
        gen_state["message"] = f"Unknown task type: {req.task_type}"
        gen_state["is_running"] = False
        gen_state["current_task"] = None
        return
        
    script_path = os.path.join(SRC_DIR, script_name)
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        import collections
        output_lines = collections.deque(maxlen=15)
        for line in process.stdout:
            output_lines.append(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            gen_state["message"] = f"{req.task_type} completed successfully."
        else:
            last_lines = '\n'.join(output_lines)
            gen_state["message"] = f"{req.task_type} failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        gen_state["message"] = f"Error during {req.task_type}: {str(e)}"
    finally:
        gen_state["is_running"] = False
        gen_state["current_task"] = None

@app.get("/api/generate/status")
def get_generate_status():
    return gen_state

@app.post("/api/generate")
def start_generation(req: GenRequest, background_tasks: BackgroundTasks):
    if gen_state["is_running"]:
        raise HTTPException(status_code=400, detail=f"Task {gen_state['current_task']} is already running.")
    background_tasks.add_task(run_generative_script, req)
    return {"message": "Generation task started."}

# --- Hyperparameter Tuning State ---
hpo_state = {
    "is_tuning": False,
    "current_model": None,
    "message": "Idle",
}

class HPOResquest(BaseModel):
    model_name: str
    num_trials: int

def run_hpo_script(req: HPOResquest):
    global hpo_state
    hpo_state["is_tuning"] = True
    hpo_state["current_model"] = req.model_name
    hpo_state["message"] = f"Starting Hyperparameter Tuning for {req.model_name} with {req.num_trials} trials...\n"
    
    script_path = os.path.join(SRC_DIR, "optuna_train_model.py")
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path, req.model_name, "--n-trials", str(req.num_trials)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        import collections
        output_lines = collections.deque(maxlen=20)
        for line in process.stdout:
            output_lines.append(line.strip())
            # Real-time streaming of the last 20 lines
            hpo_state["message"] = '\n'.join(output_lines)
            
        process.wait()
        
        if process.returncode == 0:
            last_lines = '\n'.join(output_lines)
            hpo_state["message"] = f"Tuning completed successfully for {req.model_name}.\nCheck best_hyperparameters.csv in the codebase.\n\nFinal Output:\n{last_lines}"
        else:
            last_lines = '\n'.join(output_lines)
            hpo_state["message"] = f"Tuning failed (code {process.returncode}):\n{last_lines}"
            
    except Exception as e:
        hpo_state["message"] = f"Error during tuning: {str(e)}"
    finally:
        hpo_state["is_tuning"] = False
        hpo_state["current_model"] = None

@app.get("/api/hpo/status")
def get_hpo_status():
    return hpo_state

@app.post("/api/hpo/start")
def start_hpo(req: HPOResquest, background_tasks: BackgroundTasks):
    if hpo_state["is_tuning"]:
        raise HTTPException(status_code=400, detail=f"A tuning job is already running for {hpo_state['current_model']}.")
    background_tasks.add_task(run_hpo_script, req)
    return {"message": "Hyperparameter tuning started."}

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

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_arch: str = Form("FasterRCNN"),
    model_file: str = Form(...),
    confidence: float = Form(0.5)
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    num_classes = 2
    model = get_model(model_arch, num_classes)
    model_path = os.path.join(SAVED_MODELS_DIR, model_file)
    
    model, _, _ = load_model_with_hyperparams(model, model_path, load_dir=SAVED_MODELS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((560, 480)), 
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
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
            
        img_byte = transform(img).mul(255).byte()
        label_strings = [f"polyp {s:.2f}" for s in scores.tolist()]
        drawn = draw_bounding_boxes(img_byte, boxes, labels=label_strings, colors="red", width=2)
        result_img = to_pil_image(drawn)
    else:
        # Resize original image to match tensor input format for consistency
        result_img = img.resize((480, 560))

    # Convert PIL Image to Base64
    buffered = io.BytesIO()
    result_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return PredictionResponse(boxes=result_boxes, image_base64=img_str)
