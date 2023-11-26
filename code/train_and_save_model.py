import json
import sys

from model_utils import *


def main():
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py 'model_name' '{\"BATCH_SIZE\": 2, \"LR\": 0.005, \"WEIGHT_DECAY\": 0.0005, "
              "\"CONFIDENCE_THRESHOLD\": 0.5}'")
        sys.exit(1)

    model_name = sys.argv[1]
    params = json.loads(sys.argv[2])

    debug = True
    if debug:
        max_samples = {
            'train': 20,
            'test': 10
        }
        train_dataset, test_dataset = prepare_dataset(max_samples)
    else:
        train_dataset, test_dataset = prepare_dataset()

    # Define hyperparameters
    # params = {
    #     "BATCH_SIZE": 2,
    #     "LR": 0.005,
    #     "WEIGHT_DECAY": 0.0005,
    #     "CONFIDENCE_THRESHOLD": 0.5
    # }

    # model_name = 'FasterRCNN'

    # Check device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Set number of training epochs
    num_epochs = params['NUM_EPOCHS']

    # Train the model
    trained_model, epoch_losses, batch_losses, epoch, model_path = train_model(train_dataset, params, num_epochs,
                                                                               device,
                                                                               model_s=model_name)

    # Save the trained model
    # model_path = save_model_with_hyperparams(trained_model, model_name, params, epoch_losses, batch_losses, epoch)
    # print(f"Model saved at {model_path}")


if __name__ == "__main__":
    main()
