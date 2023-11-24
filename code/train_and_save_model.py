from model_utils import *


def main():
    debug = False
    if debug:
        max_samples = {
            'train': 20,
            'test': 10
        }
        train_dataset, test_dataset = prepare_dataset(max_samples)
    else:
        train_dataset, test_dataset = prepare_dataset()

    # Define hyperparameters
    params = {
        "BATCH_SIZE": 2,
        "LR": 0.005,
        "WEIGHT_DECAY": 0.0005,
        "CONFIDENCE_THRESHOLD": 0.5
    }

    # Check device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modify the model for your specific dataset: Change the number of classes
    num_classes = 2  # 1 class + background
    model = get_model('FasterRCNN', num_classes)

    # Set number of training epochs
    num_epochs = 2

    # Train the model
    trained_model, epoch_losses, batch_losses, epoch = train_model(train_dataset, params, num_epochs, device,
                                                                   model_s='FasterRCNN')

    # Save the trained model
    model_path = save_model_with_hyperparams(trained_model, 'FasterRCNN', params, epoch_losses, batch_losses, epoch)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    main()
