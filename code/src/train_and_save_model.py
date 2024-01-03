import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(SCRIPT_DIR)

from clases.model_utils import *


def train_and_save_model():
    if len(sys.argv) < 3:
        print("Usage: python train_and_save_model.py 'model_name' '{\"BATCH_SIZE\": 2, \"LR\": 0.005, "
              "\"WEIGHT_DECAY\": 0.0005,"
              "\"CONFIDENCE_THRESHOLD\": 0.5}'")
        sys.exit(1)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = params['NUM_EPOCHS']

    trained_model, epoch_losses, batch_losses, epoch, model_path, metric_value = train_model(train_dataset, params,
                                                                                             num_epochs,
                                                                                             device,
                                                                                             model_s=model_name,
                                                                                             debug=debug)


if __name__ == "__main__":
    train_and_save_model()
