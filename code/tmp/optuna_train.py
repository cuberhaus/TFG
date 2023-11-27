import sys

import optuna

from model_utils import *


def objective(trial, train_dataset, device, model_name):
    # Generate the hyperparameters
    batch_size = trial.suggest_categorical('BATCH_SIZE', [16, 32, 64])
    lr = trial.suggest_loguniform('LR', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('WEIGHT_DECAY', 1e-10, 1e-3)

    params = {
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "WEIGHT_DECAY": weight_decay,
    }

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the model, optimizer, etc.
    model = get_model(model_name, num_classes=...)  # Define your model
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(2):  # Number of epochs can be a hyperparameter too
        model.train()
        for batch in train_loader:
            # Training steps
            # ...
            pass

    # Calculate the metric you want to optimize
    metric_to_optimize = ...  # Could be validation loss, accuracy, etc.

    return metric_to_optimize


def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py 'model_name'")
        sys.exit(1)

    model_name = sys.argv[1]

    # Check device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_dataset, test_dataset = prepare_dataset()

    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dataset, device, model_name), n_trials=100)

    # Best trial
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train the best model (optional)
    # ...


if __name__ == "__main__":
    main()
