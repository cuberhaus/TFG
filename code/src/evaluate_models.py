import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from clases.model_utils import load_model_with_hyperparams, collate_fn, prepare_dataset, parse_model_filename, \
    coco_evaluate, get_model

OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'out')
MODEL_DIR = os.path.join(OUT_DIR, 'saved_models')
CSV_FILE_PATH = os.path.join(OUT_DIR, "model_performances.csv")

parser = argparse.ArgumentParser(description="Evaluate all saved models")
parser.add_argument('--debug', action='store_true', help='Use only 5 samples for quick testing')
parser.add_argument('--models-dir', type=str, default=None,
                    help='Override saved models directory')
args = parser.parse_args()

if args.models_dir:
    MODEL_DIR = args.models_dir

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_filenames = [f for f in os.listdir(MODEL_DIR) if not f.startswith('.')]

if not model_filenames:
    print(f"No models found in {MODEL_DIR}")
    sys.exit(0)

if args.debug:
    max_samples = {'test': 5, 'train': 5}
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
    characteristics = parse_model_filename(model_filename)
    print(characteristics)

    model_name = model_filename.split('_')[0]
    model = get_model(model_name, num_classes=2)
    model, _, _ = load_model_with_hyperparams(model, model_filename, load_dir=MODEL_DIR)

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

