import os
import platform
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, \
    FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNet_ResNet50_FPN_V2_Weights, \
    retinanet_resnet50_fpn_v2
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.ops import box_iou

from custom_dataset import CustomDataset


def prepare_dataset(max_samples=None):
    # Define your data transformation (e.g., resizing, normalization, etc.)
    transform = transforms.Compose([
        transforms.Resize((560, 480)),
        transforms.ToTensor(),
    ])

    system_name = platform.system()
    test_root_dir = None
    train_root_dir = None

    if system_name == "Windows":
        train_root_dir = './data/TrainValid/TrainValid'
        test_root_dir = './data/Test/Test'
        print("Windows")
    elif system_name == "Linux":
        train_root_dir = '/home/casacuberta/TFG/TrainValid/TrainValid'
        test_root_dir = '/home/casacuberta/TFG/Test/Test'
        print("Linux")
    elif system_name == "Darwin":
        train_root_dir = '/Volumes/SSD_6Gbps/dataset1/TrainValid/TrainValid'
        test_root_dir = '/Volumes/SSD_6Gbps/dataset1/Test/Test'
        print("macOS")

    # Create instances of the custom dataset
    train_max_samples = max_samples['train'] if max_samples and 'train' in max_samples else None
    test_max_samples = max_samples['test'] if max_samples and 'test' in max_samples else None

    train_dataset = CustomDataset(root_dir=train_root_dir, transform=transform, max_samples=train_max_samples)
    test_dataset = CustomDataset(root_dir=test_root_dir, transform=transform, max_samples=test_max_samples)
    return train_dataset, test_dataset


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    """
    images, targets = zip(*batch)  # Transpose the batch (turn list of pairs into pair of lists)

    images = list(image for image in images)
    targets = list(target for target in targets)

    images = torch.stack(images, dim=0)  # Stack images to create a 4D tensor

    # In case of targets, we don't stack or pad because Faster R-CNN can handle varying-size targets
    return images, targets


def get_model(model_name, num_classes):
    """
    Get the model based on its name.
    """
    if model_name == 'FasterRCNN':
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)  # Pre-trained model
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'SSD':
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights)  # Pre-trained model
        # Adjust the number of classes for SSD
        model.head.classification_head.num_classes = num_classes
    elif model_name == 'RetinaNet':
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights)  # Pre-trained model
        in_features = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        # Replace the classifier with a new one
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_features, num_anchors=num_anchors, num_classes=num_classes)

    # elif model_name == 'MaskRCNN':
    #     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        raise Exception("Invalid model name")
    return model


def validate(model, val_loader, device):
    """
    Validate the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    return val_loss / len(val_loader)


def train_model(train_dataset, param, num_epochs, device, model_s='FasterRCNN', metric_choice='f1', debug=False):
    """
    Train the model for a specified number of epochs.
    """
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=param["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=param["BATCH_SIZE"], collate_fn=collate_fn)

    # Modify the model for your specific dataset: Change the number of classes
    # num_classes = 1 # 1 class (Polyp) + background
    num_classes = 2  # 1 class (polyp) + 1 background
    model = get_model(model_s, num_classes)

    # Define the optimizer and learning rate scheduler
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model_parameters, lr=param["LR"], momentum=0.9, weight_decay=param["WEIGHT_DECAY"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Define the loss function (this is handled by the model itself)
    # criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    print("Classes in the model's classifier:", model.roi_heads.box_predictor.cls_score.out_features)

    best_val_loss = float('inf')
    epoch_losses = []
    batch_losses = []

    # Define the directory to save models
    save_dir = 'saved_models_debug' if debug else 'saved_models'
    losses_dir = 'losses_debug' if debug else 'losses'

    saved_model_path = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        print("epoch:", epoch + 1)
        print("len(train_loader):", len(train_loader))
        for images, targets in train_loader:
            print("batch:", len(batch_losses) + 1)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss.to(device) for loss in loss_dict.values())

            epoch_loss += losses.item()
            batch_loss = losses.item()
            batch_losses.append(batch_loss)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Validation
        metrics = evaluate(model, val_loader, device)
        metric_value = metrics[metric_choice]  # Use the specified metric
        if metric_value < best_val_loss:
            best_val_loss = metric_value
            # Delete the previously saved model file, if it exists
            if saved_model_path and os.path.exists(saved_model_path):
                os.remove(saved_model_path)
                print(f"Deleted older model file: {saved_model_path}")

            saved_model_path = save_model_with_hyperparams(
                model,
                model_s,
                param,
                epoch_losses=epoch_losses,
                batch_losses=batch_losses,
                epoch=epoch,
                save_dir=save_dir,
                losses_dir=losses_dir
            )
            # torch.save(model.state_dict(), 'saved_models/best_model.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        epoch_losses.append(epoch_loss)
        lr_scheduler.step()

        for param_group in optimizer.param_groups:
            print(f'Learning Rate: {param_group["lr"]:.6f}')

    return model, epoch_losses, batch_losses, epoch, saved_model_path, metric_value


def evaluate(model, val_loader, device, iou_threshold=0.5):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    ious = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].to(device)
                pred_boxes = output['boxes'].to(device)
                scores = output['scores'].to(device)

                # Compute IoU for the predicted and ground truth boxes
                iou_matrix = box_iou(pred_boxes, gt_boxes)

                # Here, we're considering a prediction to be correct if the IoU is greater than the threshold
                # This part assumes a one-to-one matching which can be improved by using a matching strategy
                correct_preds = iou_matrix > iou_threshold

                # Now extract the IoUs for the correct predictions
                # This will give us the IoUs where the prediction was correct
                matched_ious = iou_matrix[correct_preds]

                ious.extend(matched_ious.cpu().tolist())

    # Calculate metrics based on the IoUs
    true_positives = len(ious)
    false_positives = len(outputs) - true_positives
    false_negatives = len(targets) - true_positives

    # Precision, recall, and F1 score calculations
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calculate mean IoU
    mean_iou = sum(ious) / len(ious) if ious else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ious': ious,  # List of IoU for correctly predicted boxes
        'mean_iou': mean_iou
    }


def save_model_with_hyperparams(model, model_name, hyperparams,
                                epoch_losses=None, batch_losses=None, epoch=None,
                                save_dir='./saved_models/', losses_dir='./losses/'):
    """
    Saves the model with a filename that includes the model name and hyperparameters. Losses are saved in a separate directory.

    :param epoch:
    :param batch_losses:
    :param epoch_losses:
    :param model: The model to be saved.
    :param model_name: Name of the model (string).
    :param hyperparams: Dictionary of hyperparameters.
    :param save_dir: Directory where the model will be saved.
    :param losses_dir: Directory where the losses will be saved.
    """
    # Create the parent directories if they don't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(losses_dir):
        os.makedirs(losses_dir)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Construct the filename with model and hyperparameters
    hyperparams_str = '_'.join([f'{key}-{value}' for key, value in hyperparams.items()])
    epoch_str = f'_epoch-{epoch}' if epoch is not None else ''
    base_filename = f"{model_name}_{hyperparams_str}{epoch_str}_{timestamp}"

    # Full path for the model file
    model_file_path = os.path.join(save_dir, base_filename)

    # Save the model
    torch.save(model.state_dict(), model_file_path)

    # Save epoch losses if provided
    if epoch_losses is not None:
        epoch_losses_path = os.path.join(losses_dir, f"{base_filename}_epoch_losses.txt")
        with open(epoch_losses_path, 'w') as file:
            for loss in epoch_losses:
                file.write(f"{loss}\n")
        print(f"Epoch losses saved as: {epoch_losses_path}")

    # Save batch losses if provided
    if batch_losses is not None:
        batch_losses_path = os.path.join(losses_dir, f"{base_filename}_batch_losses.txt")
        with open(batch_losses_path, 'w') as file:
            for loss in batch_losses:
                file.write(f"{loss}\n")
        print(f"Batch losses saved as: {batch_losses_path}")

    print(f"Model saved as: {model_file_path}")
    return model_file_path


def load_model_with_hyperparams(model, base_filename, load_dir='./saved_models/', losses_dir='./losses/'):
    """
    Loads the model, epoch losses, and batch losses from files.

    :param model: The model object to load the state into.
    :param base_filename: Base filename used when saving the model and losses.
    :param load_dir: Directory where the model files are stored.
    :param losses_dir: Directory where the loss files are stored.
    :return: The model, epoch_losses, and batch_losses.
    """
    # Construct file paths
    model_file_path = os.path.join(load_dir, base_filename)
    epoch_losses_path = os.path.join(losses_dir, f"{base_filename}_epoch_losses.txt")
    batch_losses_path = os.path.join(losses_dir, f"{base_filename}_batch_losses.txt")

    # Load the model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file_path))
    else:
        model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    print(f"Model loaded from: {model_file_path}")

    # Load epoch losses
    epoch_losses = []
    if os.path.exists(epoch_losses_path):
        with open(epoch_losses_path, 'r') as file:
            epoch_losses = [float(line.strip()) for line in file]
        print(f"Epoch losses loaded from: {epoch_losses_path}")

    # Load batch losses
    batch_losses = []
    if os.path.exists(batch_losses_path):
        with open(batch_losses_path, 'r') as file:
            batch_losses = [float(line.strip()) for line in file]
        print(f"Batch losses loaded from: {batch_losses_path}")

    return model, epoch_losses, batch_losses
