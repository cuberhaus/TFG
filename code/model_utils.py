import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import box_iou


def collate_fn(batch):
    images, targets = zip(*batch)  # Transpose the batch (turn list of pairs into pair of lists)

    images = list(image for image in images)
    targets = list(target for target in targets)

    images = torch.stack(images, dim=0)  # Stack images to create a 4D tensor

    # In case of targets, we don't stack or pad because Faster R-CNN can handle varying-size targets
    return images, targets


def get_model(model_name, num_classes):
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
