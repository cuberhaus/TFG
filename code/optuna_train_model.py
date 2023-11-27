import argparse

import optuna
import torch

from model_utils import train_model, prepare_dataset, evaluate
from torch.utils.data import random_split


def objective(trial, metric_to_optimize='f1', model_name='FasterRCNN'):
    # Define the hyperparameter search space using Optuna
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 10)

    # Hyperparameters to be tuned
    params = {
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "WEIGHT_DECAY": weight_decay,
        "NUM_EPOCHS": num_epochs
    }

    # Prepare your dataset
    train_dataset, _ = prepare_dataset()
    # Splitting the dataset into training and validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Train the model with the current set of hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, _, _, _, _ = train_model(train_dataset, params, num_epochs, device, model_name)

    # Evaluate the model using the evaluate function
    val_metrics = evaluate(trained_model, val_dataset, device)

    if metric_to_optimize == 'mean_iou':
        return val_metrics['mean_iou']
    elif metric_to_optimize == 'f1':
        return val_metrics['f1']
    else:
        raise ValueError(f"Unknown metric {metric_to_optimize}")


# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Run Optuna optimization.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_name', type=str, help='Name of the model to train.')
parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'mean_iou'], help='Metric to optimize.')

args = parser.parse_args()

# Create a study object and specify the optimization direction
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, metric_to_optimize=args.metric, model_name=args.model_name), n_trials=20)

# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)
