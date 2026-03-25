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
        
        # Optionally, one could read stdout here to capture logs
        process.wait()
        
        if process.returncode == 0:
            training_state["message"] = f"Training completed successfully for {req.model_name}."
        else:
            training_state["message"] = f"Training failed with return code {process.returncode}."
            
    except Exception as e:
        training_state["message"] = f"Error during training: {str(e)}"
    finally:
        training_state["is_training"] = False
        training_state["current_model"] = None

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

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
