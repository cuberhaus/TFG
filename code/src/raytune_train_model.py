import argparse
import csv

import ray
import torch
from ray import train as ray_train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Subset

from clases.model_utils import train_model, prepare_dataset


def train_model_tune(config, data_dir="./raytune", model_name='FasterRCNN', debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("Waiting for memory")
        tune.utils.wait_for_gpu()
    train_dataset, _ = prepare_dataset()
    if debug:
        subset_indices = torch.randperm(len(train_dataset))[:20]
        train_dataset = Subset(train_dataset, subset_indices)

    trained_model, _, _, _, _, metric_value = train_model(train_dataset, config, config["NUM_EPOCHS"], device,
                                                          model_name, debug=debug,
                                                          metric_choice=config["metric_choice"])

    # Free up memory
    if device.type == 'cuda':
        print("Freeing up memory")
        torch.cuda.empty_cache()

    # Inside your train_model_tune function
    ray_train.report({'metric_value': metric_value})


def tune_model(model_name, num_samples=10, max_num_epochs=10, gpus_per_trial=1, debug=False, data_dir="./raytune"):
    config = {
        "LR": tune.loguniform(1e-5, 1e-1),
        "BATCH_SIZE": tune.choice([2, 4, 8]),  # TODO: BATCH SIZE OF 16 BREAKS THINGS
        "WEIGHT_DECAY": tune.loguniform(1e-5, 1e-1),
        "NUM_EPOCHS": tune.choice(range(1, max_num_epochs + 1)),
        "metric_choice": "f1"  # or "mean_iou", depending on your needs
    }

    scheduler = ASHAScheduler(
        metric="metric_value",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = tune.CLIReporter(metric_columns=["metric_value", "training_iteration"])
    result = tune.run(
        tune.with_parameters(train_model_tune, data_dir=data_dir, model_name=model_name, debug=debug),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("metric_value", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final metric value: {}".format(best_trial.last_result["metric_value"]))

    # Save the best trial hyperparameters to a CSV file
    with open('best_hyperparameters_ray_tune.csv', 'w', newline='') as csvfile:
        fieldnames = ['parameter', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for param, value in best_trial.config.items():
            writer.writerow({'parameter': param, 'value': value})


# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Hyperparameter tuning with Ray Tune.')
parser.add_argument('model_name', type=str, help='Name of the model to train.')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for hyperparameter tuning.')
parser.add_argument('--max_num_epochs', type=int, default=10, help='Maximum number of epochs for training.')
# GPUs on mac should be = 0
parser.add_argument('--gpus_per_trial', type=int, default=1, help='GPUs per trial.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode with a smaller subset of data.')

args = parser.parse_args()

# Initialize Ray
ray.init()

# Start the tuning process
tune_model(args.model_name, args.num_samples, args.max_num_epochs, args.gpus_per_trial, args.debug,
           data_dir="./raytune")
