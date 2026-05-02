import argparse
import csv
import json
import os
import sys

import optuna
import torch


def fix_path():
    script_dir = os.path.dirname(__file__)
    relative_path = "clases"
    absolute_path = os.path.join(script_dir, relative_path)
    sys.path.append(absolute_path)


fix_path()
from clases.model_utils import train_model, prepare_dataset

try:
    import mlflow

    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("polyp-detection-optuna-sweep")
    mlflow.pytorch.autolog(log_models=False, silent=True)
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False


class _NoopCM:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _start_run(run_name: str, nested: bool = False):
    return mlflow.start_run(run_name=run_name, nested=nested) if _MLFLOW_OK else _NoopCM()


def objective(trial, metric_to_optimize='f1', model_name='FasterRCNN',
              debug=False, max_epochs=5, max_samples=None):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    if debug:
        num_epochs = 1
        ms = {'train': 8, 'test': 4}
        train_dataset, _ = prepare_dataset(ms)
    elif max_samples:
        num_epochs = trial.suggest_int("num_epochs", 1, max_epochs)
        ms = {'train': max_samples, 'test': max(max_samples // 4, 2)}
        train_dataset, _ = prepare_dataset(ms)
    else:
        num_epochs = trial.suggest_int("num_epochs", 1, max_epochs)
        train_dataset, _ = prepare_dataset()

    params = {
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "WEIGHT_DECAY": weight_decay,
        "NUM_EPOCHS": num_epochs
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _start_run(run_name=f"trial-{trial.number}", nested=True):
        if _MLFLOW_OK:
            mlflow.log_params(params)
            mlflow.log_params({
                "trial_number": trial.number,
                "model_name": model_name,
                "metric_to_optimize": metric_to_optimize,
                "debug": debug,
                "device": device.type,
            })

        trained_model, _, _, _, _, metric_value = train_model(
            train_dataset, params, num_epochs, device, model_name,
            debug=debug, metric_choice=metric_to_optimize,
        )
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if _MLFLOW_OK:
            mlflow.log_metric("metric_value", float(metric_value))

        return metric_value


def main():
    parser = argparse.ArgumentParser(description='Run Optuna optimization.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, help='Name of the model to train.')
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'mean_iou'], help='Metric to optimize.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a smaller subset of data.')
    parser.add_argument('--n-trials', type=int, default=5, help='Number of trials for Optuna optimization.')
    parser.add_argument('--max-epochs', type=int, default=5, help='Maximum number of epochs per trial.')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit total training samples per trial.')

    args = parser.parse_args()

    parent_run = _start_run(
        run_name=f"optuna-sweep-{args.model_name}-n{args.n_trials}",
        nested=False,
    )
    with parent_run:
        if _MLFLOW_OK:
            mlflow.log_params({
                "n_trials": args.n_trials,
                "max_epochs": args.max_epochs,
                "max_samples": args.max_samples,
                "metric_to_optimize": args.metric,
                "model_name": args.model_name,
            })

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, metric_to_optimize=args.metric, model_name=args.model_name,
                                    debug=args.debug, max_epochs=args.max_epochs, max_samples=args.max_samples),
            n_trials=args.n_trials)

        best_params = study.best_params
        print("Best hyperparameters: ", best_params)
        print(f"Best value (F1): {study.best_value}")

        if _MLFLOW_OK:
            mlflow.log_metric("best_value", float(study.best_value))
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'out')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'best_hyperparameters.csv'), 'w', newline='') as csvfile:
        fieldnames = ['parameter', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for param, value in best_params.items():
            writer.writerow({'parameter': param, 'value': value})

    trials_data = {
        "model_name": args.model_name,
        "n_trials": args.n_trials,
        "best_params": best_params,
        "best_value": study.best_value,
        "trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }

    with open(os.path.join(results_dir, 'hpo_results.json'), 'w') as f:
        json.dump(trials_data, f, indent=2)

    print(f"Results saved to {results_dir}/hpo_results.json")


if __name__ == "__main__":
    main()
