# Read the CSV file for plotting
import os

import pandas as pd
from matplotlib import pyplot as plt

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV file to store model performances
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "../out/model_performances.csv")
# Path to store the model performance plot
MODEL_PERFORMANCE_PLOT_PATH = os.path.join(SCRIPT_DIR, "../out/model_performance_comparison.png")

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
