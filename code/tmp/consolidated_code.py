# Code from custom_dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.annotation_paths = self.collect_paths(root_dir)

    def collect_paths(self, root_dir):
        image_paths = []
        annotation_paths = []

        annotation_root = os.path.join(root_dir, "Annotations")
        image_root = os.path.join(root_dir, "Images")

        for subdir in os.listdir(annotation_root):
            annotation_subfolder = os.path.join(annotation_root, subdir)
            image_subfolder = os.path.join(image_root, subdir)

            if os.path.isdir(annotation_subfolder) and os.path.isdir(image_subfolder):
                for filename in os.listdir(annotation_subfolder):
                    if filename.endswith(".txt"):
                        annotation_path = os.path.join(annotation_subfolder, filename)
                        image_filename = os.path.splitext(filename)[0] + ".jpg"
                        image_path = os.path.join(image_subfolder, image_filename)
                        if os.path.exists(image_path):
                            annotation_paths.append(annotation_path)
                            image_paths.append(image_path)

        return image_paths, annotation_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Load and parse annotation (you'll need to implement this part based on the content of your annotation file)
        boxes = self.parse_annotation(annotation_path)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)

        if num_objs == 0:
            # area = torch.tensor([ 0.0 ])
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }

        return image, target

    def parse_annotation(self, annotation_path):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()

        num_objects = int(lines[0].strip())
        bounding_boxes = []

        for i in range(1, num_objects + 1):
            values = list(map(int, lines[i].strip().split()))
            if len(values) == 4:
                bounding_boxes.append(values)

        return bounding_boxes


# Code from model_utils.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import box_iou


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
        model = fasterrcnn_resnet50_fpn(weights=True, pretrained=True)
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'SSD':
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        # Adjust the number of classes for SSD
        model.head.classification_head.num_classes = num_classes
    elif model_name == 'RetinaNet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
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


def train_model(train_dataset, param, num_epochs, device, model_s='FasterRCNN'):
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
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=param["LR"], momentum=0.9, weight_decay=param["WEIGHT_DECAY"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Define the loss function (this is handled by the model itself)
    # criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    print("Classes in the model's classifier:", model.roi_heads.box_predictor.cls_score.out_features)

    best_val_loss = float('inf')
    epoch_losses = []
    batch_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in train_loader:
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
        val_loss = validate(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        epoch_losses.append(epoch_loss)
        lr_scheduler.step()

        for param_group in optimizer.param_groups:
            print(f'Learning Rate: {param_group["lr"]:.6f}')

    return model, epoch_losses, batch_losses


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

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ious': ious  # List of IoU for correctly predicted boxes
    }


# Code from data_utils.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch


def get_all_bounding_boxes(dataset):
    """
    Get all bounding boxes from a dataset.
    """
    all_bounding_boxes = []
    for _, target in dataset:
        boxes = target['boxes'].cpu().numpy()
        for box in boxes:
            all_bounding_boxes.append(box)
    return all_bounding_boxes


def calculate_wcss(data, max_k=10):
    """
    Calculate the within-cluster sum of squares (WCSS) for different numbers of clusters.
    """
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss


def plot_elbow(wcss):
    """
    Plot the within-cluster sum of squares (WCSS) for different numbers of clusters.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(range(1, len(wcss) + 1))
    plt.grid(True)
    plt.show()


def calculate_silhouette_scores(data, max_k=10):
    """
    Calculate silhouette scores for different numbers of clusters.

    Parameters:
    - data: The data used for clustering.
    - max_k: Maximum number of clusters to try.

    Returns:
    - A list of silhouette scores corresponding to the number of clusters.
    """
    silhouette_scores = []
    for k in range(2, max_k + 1):  # Silhouette score is only defined for 2 or more clusters
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores


def plot_silhouette_scores(silhouette_scores):
    """
    Plot the silhouette scores for different numbers of clusters.

    Parameters:
    - silhouette_scores: List of silhouette scores.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, len(silhouette_scores) + 2))
    plt.grid(True)
    plt.show()


def cluster_bounding_boxes(bounding_boxes, n_clusters=3):
    """
    Cluster bounding boxes using K-means clustering.
    """
    # Convert from [x1, y1, x2, y2] to [x, y, width, height]
    data = []
    for bbox in bounding_boxes:
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        data.append([x, y, width, height])
    data = np.array(data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    # Get cluster centers
    centers = kmeans.cluster_centers_

    return centers, kmeans.labels_


def plot_cluster_centers(centers):
    """
    Plot the cluster centers on a chart.

    Parameters:
    - centers: numpy array of cluster centers with shape (M, 2), where M is the number of centers,
      and each center is defined by (x_center, y_center).
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Cluster Centers')

    # Annotate the cluster centers
    for i, center in enumerate(centers):
        plt.annotate(f'Center {i + 1}', (center[0], center[1]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title('Cluster Centers')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()


def plot_cluster_centers_with_bbox_centers(centers, bounding_boxes):
    """
    Plot the cluster centers and the centers of the bounding boxes on a chart.
    """

    # Convert lists to numpy arrays if they are not already
    if isinstance(bounding_boxes, list):
        bounding_boxes = np.array(bounding_boxes)
    if isinstance(centers, list):
        centers = np.array(centers)

    # Calculate the centers of the bounding boxes
    bbox_centers = np.c_[
        (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2,  # x_center
        (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2  # y_center
    ]

    # Plot the centers of the bounding boxes
    plt.scatter(bbox_centers[:, 0], bbox_centers[:, 1], c='blue', marker='o', label='Bounding Box Centers')

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Cluster Centers')

    # Annotate the cluster centers
    for i, center in enumerate(centers):
        plt.annotate(f'Cluster Center {i + 1}', (center[0], center[1]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.title('Bounding Box Centers and Cluster Centers')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()


# Code from main.ipynb
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
import torch.optim as optim
import torchvision

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from custom_dataset import CustomDataset

import os
import platform
# Define your data transformation (e.g., resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((560, 480)),
    transforms.ToTensor(),
])

train_root_dir_mac = '/Volumes/SSD_6Gbps/dataset1/TrainValid/TrainValid'
train_root_dir_windows = './data/TrainValid/TrainValid'

test_root_dir_mac = '/Volumes/SSD_6Gbps/dataset1/Test/Test'
test_root_dir_windows = './data/Test/Test'

system_name = platform.system()
train_dataset = None
test_dataset = None
if system_name == "Windows":
    print("Windows")
    # Create an instance of the custom dataset
    train_dataset = CustomDataset(root_dir=train_root_dir_windows, transform=transform)
    test_dataset = CustomDataset(root_dir=test_root_dir_windows, transform=transform)
elif system_name == "Linux":
    print("Linux")
    train_dataset = CustomDataset(root_dir=train_root_dir_windows, transform=transform)
    test_dataset = CustomDataset(root_dir=test_root_dir_windows, transform=transform)
elif system_name == "Darwin":
    print("macOS")
    train_dataset = CustomDataset(root_dir=train_root_dir_mac, transform=transform)
    test_dataset = CustomDataset(root_dir=test_root_dir_mac, transform=transform)

# Create a DataLoader to iterate through the dataset
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


max_images_to_display = 5  # Set the maximum number of images to display
image_counter = 0    # Initialize a counter

# Set the desired figure size
fig_width, fig_height = 4, 3  # Adjust these values to your preference

# Test by iterating through the dataset
for image, target in train_dataloader:
    # Convert the PyTorch tensor to a NumPy array for visualization
    image = image.squeeze(0).permute(1, 2, 0).numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.imshow(image)
    
    bounding_boxes = target['boxes']
    # Iterate through bounding boxes and draw them on the image
    if bounding_boxes.numel() > 0:
        for box in bounding_boxes:
            print(box)
            box = box.squeeze(0) # flatten the tensor to avoid unpacking error
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # Show the image with bounding boxes
    plt.show()
    
    # Increment the image counter
    image_counter += 1

    if image_counter >= max_images_to_display:
        break  # Exit the loop after displaying the desired number of images

import torch
import platform

# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check if CUDA is available, which it will be on most Windows/Linux machines with NVIDIA GPUs
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")




from model_utils import *
# Define hyperparameters
param = {
    "BATCH_SIZE":  2,
    "LR" : 0.005,
    "WEIGHT_DECAY": 0.0005,
    "CONFIDENCE_THRESHOLD": 0.5
}


# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=param["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=param["BATCH_SIZE"], collate_fn=collate_fn)


from torchvision.models.detection.retinanet import RetinaNetClassificationHead


# Modify the model for your specific dataset: Change the number of classes
# num_classes = 1 # 1 class (Polyp) + background
num_classes = 2  # 1 class (polyp) + 1 background
model = get_model('FasterRCNN', num_classes)

# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=param["LR"], momentum=0.9, weight_decay=param["WEIGHT_DECAY"])
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)
print("Classes in the model's classifier:", model.roi_heads.box_predictor.cls_score.out_features)

# For Training
images, targets = next(iter(train_loader))
images = [image.to(device) for image in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to the device

output = model(images, targets)  # Returns losses and detections
print(output)

# For inference
model.eval()

# Get a batch of images from the DataLoader
images, targets = next(iter(train_loader))

# Select the first image from the batch for inference
# Assuming your DataLoader returns a batch of images and targets
image_to_infer = images[0].unsqueeze(0).to(device)  # Add batch dimension and move to the device

# Disable gradient computation since we're only doing inference
with torch.no_grad():
    # Perform inference
    predictions = model(image_to_infer)  # Returns predictions

    # Print out the predictions for the first image
    print(predictions[0])
    
# Process the first image and its prediction
image = image_to_infer[0].cpu().numpy()  # Move the tensor to cpu and convert to numpy
image = np.transpose(image, (1, 2, 0))  # Transpose from CxHxW to HxWxC for matplotlib

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off the axis

# Add the predicted bounding boxes
for box in predictions[0]['boxes']:
    xmin, ymin, xmax, ymax = box.cpu().numpy()
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

num_epochs = 2
model, epoch_losses, batch_losses = train_model(train_dataset, param, num_epochs, device, model_s='FasterRCNN')


# After training, you can plot the epoch losses
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# Generate a timestamp or unique identifier
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Define a directory to save the models
save_dir = "./saved_models/"

# Create the parent directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the generator model and optimizer state with a unique filename
generator_checkpoint_file = f"{save_dir}generator_checkpoint_{timestamp}.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': model.state_dict()
}, generator_checkpoint_file)

# Define filenames with timestamps
epoch_losses_filename =  os.path.join(save_dir, f"epoch_losses_{timestamp}.txt")
batch_losses_filename =  os.path.join(save_dir, f"batch_losses_{timestamp}.txt")
 
# Save to a file with a timestamp
with open(epoch_losses_filename, 'w') as file:
    for loss in epoch_losses:
        file.write(f"{loss}\n")
        
with open(batch_losses_filename, 'w') as file:
    for loss in batch_losses:
        file.write(f"{loss}\n")

# Provide the timestamp you want to load
specified_timestamp = "20231022200319"  # Replace with the desired timestamp

# Load the generator model and optimizer state
checkpoint = torch.load(f'./saved_models/generator_checkpoint_{specified_timestamp}.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint['optimizer_state_dict'])

# Define the filenames based on the specified timestamp
specified_epoch_losses_filename = f"./saved_models/epoch_losses_{specified_timestamp}.txt"
specified_batch_losses_filename = f"./saved_models/batch_losses_{specified_timestamp}.txt"

# Define lists to store the loaded losses
loaded_epoch_losses = []
loaded_batch_losses = []

# Load G_losses from the file with the specified timestamp
with open(specified_batch_losses_filename, 'r') as file:
    for line in file:
        loaded_batch_losses.append(float(line.strip()))

# Load D_losses from the file with the specified timestamp
with open(specified_epoch_losses_filename, 'r') as file:
    for line in file:
        loaded_epoch_losses.append(float(line.strip()))

epoch_losses = loaded_epoch_losses
batch_losses = loaded_batch_losses

from torchvision.ops import box_iou

# Usage:
metrics = evaluate(model, val_loader, device, params["CONFIDENCE_THRESHOLD"])
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1']}")
print(f"Average IoU: {sum(metrics['ious']) / len(metrics['ious']) if metrics['ious'] else 0}")


from data_utils import *

# This can take a while to run!
# Assuming train_dataset is an instance of CustomDataset
all_bounding_boxes = get_all_bounding_boxes(train_dataset)
print("Number of bounding boxes:", len(all_bounding_boxes))

# Calculate WCSS for a range of number of clusters
wcss = calculate_wcss(all_bounding_boxes, max_k=10)
plot_elbow(wcss)

from sklearn.metrics import silhouette_score

# Calculate silhouette scores for a range of number of clusters
silhouette_scores = calculate_silhouette_scores(all_bounding_boxes, max_k=10)
plot_silhouette_scores(silhouette_scores)

# Use the function with your data
centers_8, labels_8 = cluster_bounding_boxes(all_bounding_boxes,8)
centers_3, labels_3 = cluster_bounding_boxes(all_bounding_boxes,3)
print("Cluster Centers:", centers_8)

# Example usage
# Assume that `centers` contains the cluster centers calculated from your bounding boxes.
plot_cluster_centers(centers_8)
plot_cluster_centers(centers_3)


# Example usage
# Assume that `centers` contains the cluster centers calculated from your bounding boxes.
# Also, assume that `bounding_boxes` contains your bounding box data.
plot_cluster_centers_with_bbox_centers(centers_3, all_bounding_boxes)
plot_cluster_centers_with_bbox_centers(centers_8, all_bounding_boxes)



# # Importar las bibliotecas necesarias
# import torch
# from torch import nn
# from torch.optim import Adam
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from cycle_gan_model import Generator, Discriminator  # Suponiendo que tienes implementaciones de Generator y Discriminator
# 
# # Definir transformaciones para las imágenes
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Cambiar al tamaño deseado
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# 
# # Cargar los datasets
# # Supongamos que tienes dos carpetas: '/path/to/masks' para máscaras y '/path/to/real_images' para imágenes reales
# dataset_masks = ImageFolder('/path/to/masks', transform=transform)
# dataset_real_images = ImageFolder('/path/to/real_images', transform=transform)
# 
# # Crear DataLoaders para los datasets
# loader_masks = DataLoader(dataset_masks, batch_size=1, shuffle=True)
# loader_real_images = DataLoader(dataset_real_images, batch_size=1, shuffle=True)
# 
# # Inicializar los modelos GAN
# G_mask_to_image = Generator()  # Generador que convierte máscaras en imágenes
# G_image_to_mask = Generator()  # Generador que convierte imágenes en máscaras
# D_image = Discriminator()  # Discriminador para imágenes
# D_mask = Discriminator()  # Discriminador para máscaras
# 
# # Mover modelos al dispositivo adecuado (CPU o GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# G_mask_to_image.to(device)
# G_image_to_mask.to(device)
# D_image.to(device)
# D_mask.to(device)
# 
# # Definir los optimizadores para los generadores y discriminadores
# optimizer_G = Adam(list(G_mask_to_image.parameters()) + list(G_image_to_mask.parameters()), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D_image = Adam(D_image.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D_mask = Adam(D_mask.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 
# # Definir la función de pérdida (adversarial loss)
# criterion_GAN = nn.MSELoss()
# criterion_cycle = nn.L1Loss()  # Para la consistencia cíclica
# criterion_identity = nn.L1Loss()  # Para la pérdida de identidad (opcional)
# 
# # Bucle de entrenamiento
# for epoch in range(num_epochs):
#     for mask, real_image in zip(loader_masks, loader_real_images):
# 
#         # Preparar los inputs
#         real_image = real_image.to(device)
#         mask = mask.to(device)
# 
#         # Generar una imagen falsa a partir de la máscara
#         fake_image = G_mask_to_image(mask)
# 
#         # Entrenar los discriminadores (D_image y D_mask)
#         # ...
# 
#         # Calcular la pérdida para los generadores (G_mask_to_image y G_image_to_mask)
#         # ...
# 
#         # Retropropagación y optimización para los generadores
#         # ...
# 
#         # Retropropagación y optimización para los discriminadores
#         # ...
# 
# # Guardar los modelos entrenados
# torch.save(G_mask_to_image.state_dict(), 'G_mask_to_image.pth')
# torch.save(G_image_to_mask.state_dict(), 'G_image_to_mask.pth')
# torch.save(D_image.state_dict(), 'D_image.pth')
# torch.save(D_mask.state_dict(), 'D_mask.pth')
# 
# # Generar imágenes para evaluación o más entrenamiento
# G_mask_to_image.eval()
# with torch.no_grad():
#     for mask in loader_masks:
#         mask = mask.to(device)
#         synthetic_image = G_mask_to_image(mask)
#         # Guardar o utilizar synthetic_image


# Train the model
# epoch_losses = []  # List to store epoch loss
# batch_losses = []  # List to store losses for each batch
# 
# best_val_loss = float('inf')  # Initialize best validation loss
# 
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0  # Keep track of the total loss for this epoch
#     for images, targets in train_loader:
#         # Move images and targets to device
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
#        
#         # Debugging code to check the device of the inputs 
#         # print(f"Device of model parameters: {[p.device for p in model.parameters()]}")
#         # print(f"Device of images: {images[0].device}")
#         # print(f"Device of targets: {targets[0]['boxes'].device}")
# 
#         loss_dict = model(images, targets)
#         losses = sum(loss.to(device) for loss in loss_dict.values())
#         
#         epoch_loss += losses.item()  # Add the batch loss to the total epoch loss
# 
#         # Record the loss for the current batch
#         batch_loss = losses.item()
#         batch_losses.append(batch_loss)
#         
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#         
#     # After training for an epoch, validate
#     val_loss = validate(model, val_loader, device)
#     
#     # Check if this is the best model based on validation loss
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         # Save the best model
#         torch.save(model.state_dict(), 'best_model.pth')
# 
#     # Print out the information
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
#     # Append the epoch loss to the list
#     epoch_losses.append(epoch_loss)
#     # Update the learning rate
#     lr_scheduler.step()
#     # Print learning rate for each parameter group
#     for param_group in optimizer.param_groups:
#         print(f'Learning Rate: {param_group["lr"]:.6f}')

