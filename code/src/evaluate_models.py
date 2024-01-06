import os
import platform

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from clases.model_utils import load_model_with_hyperparams, collate_fn, prepare_dataset, parse_model_filename, \
    coco_evaluate

os_name = platform.system()

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = None
TEST_DATA_PATH = None
if os_name == 'Linux':
    TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '../data/dataset1/Test/Test/')
    MODEL_DIR = os.path.join(SCRIPT_DIR, '../../old/saved_models/')  # Remote
elif os_name == 'Darwin':
    MODEL_DIR = os.path.join(SCRIPT_DIR, '../tmp/saved_models/')  # Mac
    TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '/Volumes/SSD_6Gbps/dataset1/Test/Test/')  # Mac
elif os_name == 'Windows':
    MODEL_DIR = os.path.join(SCRIPT_DIR, '../tmp/saved_models/')  # Remote
    TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '../data/Test/Test/')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "../out/model_performances.csv")
MODEL_PERFORMANCE_PLOT_PATH = os.path.join(SCRIPT_DIR, "../out/model_performance_comparison.png")
DEBUG = True


model_filenames = [f for f in os.listdir(MODEL_DIR)]

if DEBUG:
    max_samples = {
        'test': 5,
        'train': 5
    }
else:
    max_samples = None

train_dataset, test_dataset = prepare_dataset(max_samples)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(CSV_FILE_PATH):
    os.remove(CSV_FILE_PATH)

# Check if the CSV file exists
csv_exists = os.path.exists(CSV_FILE_PATH)

for model_filename in model_filenames:
    print(f"Evaluating {model_filename}")
    # Parse model characteristics
    characteristics = parse_model_filename(model_filename)
    print(characteristics)

    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)  # Update num_classes as per your requirement
    full_model_path = os.path.join(MODEL_DIR, model_filename)
    model, _, _ = load_model_with_hyperparams(model, full_model_path, load_dir=MODEL_DIR)

    # Evaluate the model
    model.to(device)
    metrics = coco_evaluate(model, test_loader, device)
    print(f"Metrics: {metrics}")

    # Append performance metrics and characteristics to the DataFrame and write to CSV
    performance_data = {
        **characteristics,  # Unpack parsed characteristics
        'AP_50_95_all': metrics[0],
        'AP_50_all': metrics[1],
        'AP_75_all': metrics[2],
        'AP_50_95_small': metrics[3],
        'AP_50_95_medium': metrics[4],
        'AP_50_95_large': metrics[5],
        'AR_50_95_all_maxDets_1': metrics[6],
        'AR_50_95_all_maxDets_10': metrics[7],
        'AR_50_95_all_maxDets_100': metrics[8],
        'AR_50_95_small_maxDets_100': metrics[9],
        'AR_50_95_medium_maxDets_100': metrics[10],
        'AR_50_95_large_maxDets_100': metrics[11]
    }
    print(f"Performance data: {performance_data}")
    # If CSV does not exist, write with header, else append without header
    if not csv_exists:
        pd.DataFrame([performance_data]).to_csv(CSV_FILE_PATH, mode='a', header=True, index=False)
        csv_exists = True  # Update the flag so headers are not added again
    else:
        pd.DataFrame([performance_data]).to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)

