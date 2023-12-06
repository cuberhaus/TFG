import argparse
import json
import sys

import os


def fix_path():
    script_dir = os.path.dirname(__file__)  # Directory of the script file
    relative_path = "clases"  # Relative path to the file
    absolute_path = os.path.join(script_dir, relative_path)  # Full path to the file
    print(absolute_path)
    sys.path.append(absolute_path)


fix_path()

from clases.model_utils import *


def train_and_save_model():
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py 'model_name' '{\"BATCH_SIZE\": 2, \"LR\": 0.005, \"WEIGHT_DECAY\": 0.0005, "
              "\"CONFIDENCE_THRESHOLD\": 0.5}'")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and save model.')
    parser.add_argument('model_name', type=str, help='Name of the model to train.')
    parser.add_argument('params', type=json.loads, help='Hyperparameter settings in JSON format.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    args = parser.parse_args()

    model_name = args.model_name
    params = args.params
    debug = args.debug

    if debug:
        max_samples = {
            'train': 20,
            'test': 10
        }
        train_dataset, test_dataset = prepare_dataset(max_samples)
    else:
        train_dataset, test_dataset = prepare_dataset()

    # Check device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set number of training epochs
    num_epochs = params['NUM_EPOCHS']

    # Train the model
    trained_model, epoch_losses, batch_losses, epoch, model_path, metric_value = train_model(train_dataset, params,
                                                                                             num_epochs,
                                                                                             device,
                                                                                             model_s=model_name,
                                                                                             debug=debug)


if __name__ == "__main__":
    train_and_save_model()
