import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(SCRIPT_DIR)

import torch
from clases.model_utils import prepare_dataset, train_model

try:
    import mlflow

    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("polyp-detection-baseline")
    mlflow.pytorch.autolog(log_models=False, silent=True)
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False


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
    parser.add_argument('--max-samples', type=int, default=None, help='Limit total training+test samples.')

    args = parser.parse_args()

    model_name = args.model_name
    params = args.params
    debug = args.debug

    if args.max_samples:
        n = args.max_samples
        max_samples = {'train': n, 'test': max(n // 4, 2)}
        train_dataset, test_dataset = prepare_dataset(max_samples)
    elif debug:
        max_samples = {'train': 8, 'test': 4}
        train_dataset, test_dataset = prepare_dataset(max_samples)
    else:
        train_dataset, test_dataset = prepare_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = params['NUM_EPOCHS']

    run_cm = mlflow.start_run(run_name=f"baseline-{model_name}") if _MLFLOW_OK else _NoopCM()
    with run_cm:
        if _MLFLOW_OK:
            mlflow.log_params(params)
            mlflow.log_params({"model_name": model_name, "debug": debug, "device": device.type})
            if args.max_samples:
                mlflow.log_param("max_samples", args.max_samples)

        trained_model, epoch_losses, batch_losses, epoch, model_path, metric_value = train_model(
            train_dataset, params, num_epochs, device, model_s=model_name, debug=debug
        )

        if _MLFLOW_OK:
            mlflow.log_metric("metric_value", float(metric_value))
            mlflow.log_metric("epochs_completed", int(epoch))
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")


class _NoopCM:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    train_and_save_model()
