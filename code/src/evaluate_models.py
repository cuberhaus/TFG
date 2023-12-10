import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torch

from clases.custom_dataset import CustomDataset
from clases.model_utils import load_model_with_hyperparams, evaluate, collate_fn, prepare_dataset

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory containing the saved models
# MODEL_DIR = os.path.join(SCRIPT_DIR, '../tmp/saved_models/') # Mac
MODEL_DIR = os.path.join(SCRIPT_DIR, '../../old/saved_models/')  # Remote
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print(MODEL_DIR)
# CSV file to store model performances
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "../out/model_performances.csv")
# Path to the test data
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '../data/dataset1/Test/Test/')
# TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '/Volumes/SSD_6Gbps/dataset1/Test/Test/') # Mac
# Path to store the model performance plot
MODEL_PERFORMANCE_PLOT_PATH = os.path.join(SCRIPT_DIR, "../out/model_performance_comparison.png")
# Set to True to run in debug mode
DEBUG = False


# Function to check if the filename matches the model naming pattern
def is_model_file(filename):
    return filename.startswith('FasterRCNN_')


# Function to parse model filename and extract characteristics
def parse_model_filename(filename):
    parts = filename.replace('.pth', '').split('_')

    model_type = parts[0]  # The first part of the filename is the model type
    characteristics = {'Model': model_type}

    for part in parts[1:]:  # Start from the second part as the first is the model type
        if '-' in part:
            key, value = part.split('-', 1)
            characteristics[key] = value
    return characteristics


# List all files in the model directory and filter out non-model files
model_filenames = [f for f in os.listdir(MODEL_DIR) if is_model_file(f)]
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

# Initialize DataFrame columns if the file doesn't exist
if not os.path.exists(CSV_FILE_PATH):
    pd.DataFrame(columns=['Model', 'Batch Size', 'Learning Rate', 'Weight Decay', 'Num Epochs', 'Precision', 'Recall',
                          'F1-Score', 'Mean IoU']).to_csv(CSV_FILE_PATH, index=False)

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
    metrics = evaluate(model, test_loader, device)
    print(f"Metrics: {metrics}")

    # Append performance metrics and characteristics to the DataFrame and write to CSV
    performance_data = {
        'Model': model_filename,
        **characteristics,  # Unpack parsed characteristics
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'Mean IoU': metrics['mean_iou']
    }
    pd.DataFrame([performance_data]).to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)

# Read the CSV file for plotting
model_performances = pd.read_csv(CSV_FILE_PATH)

# Plotting performance metrics
plt.figure(figsize=(12, 8))
for metric in ['Precision', 'Recall', 'F1-Score', 'Mean IoU']:
    plt.plot(model_performances['Model'], model_performances[metric], marker='o', label=metric)

plt.xlabel('Model')
plt.ylabel('Performance')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_PERFORMANCE_PLOT_PATH)
plt.show()
