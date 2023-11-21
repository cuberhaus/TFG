
import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from model_utils import get_model, train_model, save_model_with_hyperparams
from torchvision import transforms
import os
from model_utils import *

def main():
    # Define dataset directories (modify as per your paths)
    # train_root_dir = './data/TrainValid/TrainValid'  # Replace with your training dataset directory

    train_dataset, test_dataset = prepare_dataset()

    # Create an instance of the custom dataset
    # train_dataset = CustomDataset(root_dir=train_root_dir, transform=transform)

    # Define hyperparameters
    params = {
        "BATCH_SIZE":  2,
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

    # Move model to the correct device
    model = model.to(device)

    # Set number of training epochs
    num_epochs = 2

    # Train the model
    trained_model, epoch_losses, batch_losses, epoch = train_model(train_dataset, params, num_epochs, device, model_s='FasterRCNN')

    # Save the trained model
    model_path = save_model_with_hyperparams(trained_model, 'FasterRCNN', params, epoch_losses, batch_losses, epoch)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
