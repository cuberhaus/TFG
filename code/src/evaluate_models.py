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
    # Directory containing the saved models
    MODEL_DIR = os.path.join(SCRIPT_DIR, '../tmp/saved_models/')  # Mac
    # Path to the test data
    TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '/Volumes/SSD_6Gbps/dataset1/Test/Test/')  # Mac

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print(MODEL_DIR)
# CSV file to store model performances
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "../out/model_performances.csv")
# Path to store the model performance plot
MODEL_PERFORMANCE_PLOT_PATH = os.path.join(SCRIPT_DIR, "../out/model_performance_comparison.png")
# Set to True to run in debug mode
DEBUG = True


# List all files in the model directory and filter out non-model files
model_filenames = [f for f in os.listdir(MODEL_DIR)]
print(model_filenames)

if DEBUG:
    max_samples = {
        'test': 5,
        'train': 5
    }
else:
    max_samples = None

# Prepare your dataset
train_dataset, test_dataset = prepare_dataset(max_samples)  # Update with your actual dataset
# test_dataset = CustomDataset(root_dir=TEST_DATA_PATH,
#                              transform=None)  # Update with your actual path and transforms
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(CSV_FILE_PATH):
    pd.DataFrame(columns=[
        'Model', 'Batch Size', 'Learning Rate', 'Weight Decay', 'Num Epochs', 'Best_Epoch',
        'AP_50_95_all', 'AP_50_all', 'AP_75_all',
        'AP_50_95_small', 'AP_50_95_medium', 'AP_50_95_large',
        'AR_50_95_all_maxDets_1', 'AR_50_95_all_maxDets_10', 'AR_50_95_all_maxDets_100',
        'AR_50_95_small_maxDets_100', 'AR_50_95_medium_maxDets_100', 'AR_50_95_large_maxDets_100'
    ]).to_csv(CSV_FILE_PATH, index=False)

# # Initialize DataFrame columns if the file doesn't exist
# if not os.path.exists(CSV_FILE_PATH):
#     pd.DataFrame(columns=['Model', 'Batch Size', 'Learning Rate', 'Weight Decay', 'Num Epochs', 'Precision', 'Recall',
#                           'F1-Score', 'Mean IoU']).to_csv(CSV_FILE_PATH, index=False)

for model_filename in model_filenames:
    print(f"Evaluating {model_filename}")
    # Parse model characteristics
    characteristics = parse_model_filename(model_filename)

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
    pd.DataFrame([performance_data]).to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)

