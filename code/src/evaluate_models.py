import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torch

from clases.custom_dataset import CustomDataset
from clases.model_utils import load_model_with_hyperparams, evaluate, collate_fn

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory containing the saved models
MODEL_DIR = os.path.join(SCRIPT_DIR, '../tmp/saved_models/')
# CSV file to store model performances
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "../out/model_performances.csv")
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, '../data/test/')

# Function to check if the filename matches the model naming pattern
def is_model_file(filename):
    return filename.startswith('FasterRCNN_') and filename.endswith('.pth')


# Function to parse model filename and extract characteristics
def parse_model_filename(filename):
    parts = filename.replace('.pth', '').split('_')
    characteristics = {part.split('-')[0]: part.split('-')[1] for part in parts if '-' in part}
    return characteristics


# List all files in the model directory and filter out non-model files
model_filenames = [f for f in os.listdir(MODEL_DIR) if is_model_file(f)]

# Prepare your dataset
test_dataset = CustomDataset(root_dir='path_to_test_data',
                             transform=None)  # Update with your actual path and transforms
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize DataFrame columns if the file doesn't exist
if not os.path.exists(CSV_FILE_PATH):
    pd.DataFrame(columns=['Model', 'Batch Size', 'Learning Rate', 'Weight Decay', 'Num Epochs', 'Precision', 'Recall',
                          'F1-Score', 'Mean IoU']).to_csv(CSV_FILE_PATH, index=False)

for model_filename in model_filenames:
    # Parse model characteristics
    characteristics = parse_model_filename(model_filename)

    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)  # Update num_classes as per your requirement
    full_model_path = os.path.join(MODEL_DIR, model_filename)
    model, _, _ = load_model_with_hyperparams(model, full_model_path, load_dir=MODEL_DIR)

    # Evaluate the model
    model.to(device)
    metrics = evaluate(model, test_loader, device)

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
plt.savefig('model_performance_comparison.png')
plt.show()
