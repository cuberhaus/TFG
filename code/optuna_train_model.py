import argparse
import csv

import optuna
import torch

from model_utils import train_model, prepare_dataset, evaluate
from torch.utils.data import random_split, Subset


def objective(trial, metric_to_optimize='f1', model_name='FasterRCNN', debug=False):
    # Define the hyperparameter search space using Optuna
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])  # TODO: BATCH SIZE OF 16 BREAKS THINGS
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 5)   # TODO: more epochs when we have more time

    # Hyperparameters to be tuned
    params = {
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "WEIGHT_DECAY": weight_decay,
        "NUM_EPOCHS": num_epochs
    }

    # Prepare your dataset
    train_dataset, _ = prepare_dataset()
    if debug:
        # Use a smaller subset of the dataset for debugging
        subset_indices = torch.randperm(len(train_dataset))[:20]  # Adjust the size as needed
        train_dataset = Subset(train_dataset, subset_indices)
    else:
        train_dataset = train_dataset

    # Train the model with the current set of hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Validation split is done inside the train_model function already
    trained_model, _, _, _, _, metric_to_optimize = train_model(train_dataset, params, num_epochs, device, model_name,
                                                                debug=debug, metric_choice=metric_to_optimize)

    # Free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return metric_to_optimize


# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Run Optuna optimization.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_name', type=str, help='Name of the model to train.')
parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'mean_iou'], help='Metric to optimize.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode with a smaller subset of data.')

args = parser.parse_args()

# Create a study object and specify the optimization direction
study = optuna.create_study(direction='maximize')
study.optimize(
    lambda trial: objective(trial, metric_to_optimize=args.metric, model_name=args.model_name, debug=args.debug),
    n_trials=5) # TODO: more trials when we have more time

# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)

# Write the best hyperparameters to a CSV file
with open('best_hyperparameters.csv', 'w', newline='') as csvfile:
    fieldnames = ['parameter', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for param, value in best_params.items():
        writer.writerow({'parameter': param, 'value': value})
