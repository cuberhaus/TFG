import json
import os
from datetime import datetime
import re

from PIL import Image
import torch
import torch.optim as optim
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, \
    FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights, \
    retinanet_resnet50_fpn_v2, RetinaNetHead
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.ops import box_iou

from .custom_dataset import CustomDataset
from .utils import *

SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(SCRIPT_DIR, '../')
PROJ_DIR = os.path.join(SRC_DIR, '../')
OUT_DIR = os.path.join(SRC_DIR, '..', 'out')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def prepare_dataset(max_samples=None):
    transform = transforms.Compose([
        transforms.Resize((560, 480)),
        transforms.ToTensor(),
    ])

    test_root_dir, train_root_dir = dataset_paths()

    train_max_samples = max_samples['train'] if max_samples and 'train' in max_samples else None
    test_max_samples = max_samples['test'] if max_samples and 'test' in max_samples else None

    train_dataset = CustomDataset(root_dir=train_root_dir, transform=transform, max_samples=train_max_samples)
    test_dataset = CustomDataset(root_dir=test_root_dir, transform=transform, max_samples=test_max_samples)
    return train_dataset, test_dataset


def dataset_paths():
    system_name = platform.system()
    test_root_dir = None
    train_root_dir = None
    if system_name == "Windows":
        train_root_dir = os.path.join(PROJ_DIR, 'data/TrainValid/TrainValid')
        test_root_dir = os.path.join(PROJ_DIR, 'data/Test/Test')
        print("Windows")
    elif is_wsl():
        # train_root_dir = '/mnt/c/Users/polcg/repos/TFG/code/data/TrainValid/TrainValid'
        # test_root_dir = '/mnt/c/Users/polcg/repos/TFG/code/data/Test/Test'
        train_root_dir = os.path.join(PROJ_DIR, 'data/TrainValid/TrainValid')
        test_root_dir = os.path.join(PROJ_DIR, 'data/Test/Test')
        print("wsl")
    elif system_name == "Linux":
        train_root_dir = os.path.join(PROJ_DIR, 'data/TrainValid/TrainValid')
        test_root_dir = os.path.join(PROJ_DIR, 'data/Test/Test')
        # train_root_dir = '/home/casacuberta/TFG/TrainValid/TrainValid'
        # test_root_dir = '/home/casacuberta/TFG/Test/Test'
        print("Linux")
    elif system_name == "Darwin":
        train_root_dir = '/Volumes/SSD_6Gbps/dataset1/TrainValid/TrainValid'
        test_root_dir = '/Volumes/SSD_6Gbps/dataset1/Test/Test'
        print("macOS")
    print("train dir: " + train_root_dir)
    print("test dir: " + test_root_dir)

    return test_root_dir, train_root_dir


def collate_fn(batch):
    images, targets = zip(*batch)

    images = list(image for image in images)
    targets = list(target for target in targets)

    images = torch.stack(images, dim=0)

    return images, targets


def get_model(model_name, num_classes):
    if model_name == 'FasterRCNN':
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'SSD':
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        model.head.classification_head.num_classes = num_classes
    elif model_name == 'RetinaNet':
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head = RetinaNetHead(in_channels, num_classes=num_classes, num_anchors=num_anchors)
    else:
        raise Exception("Invalid model name")
    return model


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    return val_loss / len(val_loader)


def train_model(train_dataset, param, num_epochs, device, model_s='FasterRCNN', metric_choice='f1', debug=False):
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=param["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=param["BATCH_SIZE"], collate_fn=collate_fn)

    num_classes = 2  # 1 class (polyp) + 1 background
    model = get_model(model_s, num_classes)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model_parameters, lr=param["LR"], momentum=0.9, weight_decay=param["WEIGHT_DECAY"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # Loss function is handled by the model

    model = model.to(device)

    best_val_loss = float('inf')
    epoch_losses = []
    batch_losses = []

    save_dir = os.path.join(OUT_DIR, 'saved_models_debug' if debug else 'saved_models')
    print("save_dir:", save_dir)
    losses_dir = os.path.join(OUT_DIR, 'losses_debug' if debug else 'losses')
    print("losses_dir:", losses_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(losses_dir):
        os.makedirs(losses_dir)

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

        metrics = coco_evaluate(model, val_loader, device)
        AP_score = metrics[0]
        AR_score = metrics[8]
        f1_score = 2 * (AP_score * AR_score) / (AP_score + AR_score)
        metric_value = f1_score

        if metric_value > best_val_loss:
            print(str(metric_choice) + " : " + str(metric_value))
            best_val_loss = metric_value
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        epoch_losses.append(epoch_loss)
        lr_scheduler.step()

        for param_group in optimizer.param_groups:
            print(f'Learning Rate: {param_group["lr"]:.6f}')

    return model, epoch_losses, batch_losses, epoch, saved_model_path, metric_value


def convert_to_coco_format(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def extract_image_info(custom_dataset):
    images_info = []
    for idx, img_path in enumerate(custom_dataset.image_paths):
        with Image.open(img_path) as img:
            width, height = img.size
            image_info = {
                "id": idx,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height
            }
            images_info.append(image_info)
    return images_info


def create_default_predictions(image_ids):
    default_predictions = []
    for image_id in image_ids:
        default_pred = {
            "image_id": image_id,
            "category_id": 0,  # or -1
            "bbox": [0, 0, 0, 0],
            "score": 0.0
        }
        default_predictions.append(default_pred)
    return default_predictions


def coco_evaluate(model, val_loader, device, iou_threshold=0.5):
    predictions_dir = os.path.join(OUT_DIR, 'predictions.json')
    ground_truth_dir = os.path.join(OUT_DIR, 'ground_truth.json')
    model.eval()
    # Initialize dataset for COCO ground truth object
    coco_gt_dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    coco_gt = COCO()  # Initialize COCO ground truth object
    coco_gt.dataset = coco_gt_dataset  # Load dataset into COCO ground truth object

    coco_dt = []  # List to store predictions in COCO format

    categories = [{'id': 1, 'name': 'polyp'}]  # Define categories
    coco_gt.dataset['categories'] = categories  # Load categories into COCO ground truth object

    ann_id = 0
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for target, output, image in zip(targets, outputs, images):
            image_id = target['image_id'].item()

            width, height = image.size()[2], image.size()[1]
            file_name = f"image_{image_id}.jpg"

            image_info = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height
            }
            coco_gt.dataset['images'].append(image_info)

            # Convert ground truth to COCO format
            for box, label in zip(target['boxes'], target['labels']):
                box = convert_to_coco_format(box.tolist())
                coco_gt_annotation = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': label.item(),
                    'area': (box[2] * box[3]),
                    'bbox': box,
                    'iscrowd': 0
                }
                coco_gt.dataset['annotations'].append(coco_gt_annotation)

            # Convert predictions to COCO format
            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                box = box.detach().cpu().numpy().tolist()
                box = convert_to_coco_format(box)
                coco_dt_annotation = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': box,
                    'score': score.item()
                }
                coco_dt.append(coco_dt_annotation)
                ann_id += 1

    # # Save predictions to a JSON file
    with open(predictions_dir, 'w') as f:
        json.dump(coco_dt, f)

    with open(ground_truth_dir, 'w') as f:
        json.dump(coco_gt.dataset, f)

    # Load the ground truth
    cocoGt = COCO(ground_truth_dir)

    # Assuming predictions is a list of your model's predictions
    if len(coco_dt) == 0:
        image_ids = [img['id'] for img in cocoGt.imgs.values()]
        predictions = create_default_predictions(image_ids)
        coco_dt = predictions
        cocoDt = cocoGt.loadRes(coco_dt)
    else:
        # Load the predictions
        cocoDt = cocoGt.loadRes(predictions_dir)

    # Initialize COCOeval object
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # Evaluate on the data
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats


# FIXME: evaluate
# def evaluate(model, val_loader, device, iou_threshold=0.5):
#     model.eval()
#     ious = []
#     image_counter = 0
#     last_printed = 0
#     multiple = 100
#
#     with torch.no_grad():
#         for images, targets in val_loader:
#             image_counter += len(images)  # Update the counter with the batch size
#             # Check if the counter has passed a multiple of 100 since the last print
#             if image_counter // multiple > last_printed:
#                 print(f"Processed {image_counter} images...")
#                 last_printed = image_counter // multiple  # Update the last printed multiple
#
#             images = list(img.to(device) for img in images)
#             outputs = model(images)
#
#             for target, output in zip(targets, outputs):
#                 gt_boxes = target['boxes'].to(device)
#                 pred_boxes = output['boxes'].to(device)
#                 scores = output['scores'].to(device)
#
#                 # Compute IoU for the predicted and ground truth boxes
#                 iou_matrix = box_iou(pred_boxes, gt_boxes)
#
#                 # Here, we're considering a prediction to be correct if the IoU is greater than the threshold
#                 # This part assumes a one-to-one matching which can be improved by using a matching strategy
#                 correct_preds = iou_matrix > iou_threshold
#
#                 # Now extract the IoUs for the correct predictions
#                 # This will give us the IoUs where the prediction was correct
#                 matched_ious = iou_matrix[correct_preds]
#
#                 ious.extend(matched_ious.cpu().tolist())
#
#     true_positives = len(ious)
#     false_positives = len(outputs) - true_positives
#     false_negatives = len(targets) - true_positives
#
#     precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
#     recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#
#     mean_iou = sum(ious) / len(ious) if ious else 0
#
#     return {
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'ious': ious,
#         'mean_iou': mean_iou
#     }


def save_model_with_hyperparams(model, model_name, hyperparams,
                                epoch_losses=None, batch_losses=None, epoch=None,
                                save_dir='./out/saved_models/', losses_dir='./out/losses/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(losses_dir):
        os.makedirs(losses_dir)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    hyperparams_str = '_'.join([f'{key}-{value}' for key, value in hyperparams.items()])
    epoch_str = f'_epoch-{epoch}' if epoch is not None else ''
    base_filename = f"{model_name}_{hyperparams_str}{epoch_str}_{timestamp}"

    model_file_path = os.path.join(save_dir, base_filename)

    torch.save(model.state_dict(), model_file_path)

    if epoch_losses is not None:
        epoch_losses_path = os.path.join(losses_dir, f"{base_filename}_epoch_losses.txt")
        with open(epoch_losses_path, 'w') as file:
            for loss in epoch_losses:
                file.write(f"{loss}\n")
        print(f"Epoch losses saved as: {epoch_losses_path}")

    if batch_losses is not None:
        batch_losses_path = os.path.join(losses_dir, f"{base_filename}_batch_losses.txt")
        with open(batch_losses_path, 'w') as file:
            for loss in batch_losses:
                file.write(f"{loss}\n")
        print(f"Batch losses saved as: {batch_losses_path}")

    print(f"Model saved as: {model_file_path}")
    return model_file_path


def load_model_with_hyperparams(model, base_filename, load_dir='./saved_models/', losses_dir='./losses/'):
    model_file_path = os.path.join(load_dir, base_filename)
    epoch_losses_path = os.path.join(losses_dir, f"{base_filename}_epoch_losses.txt")
    batch_losses_path = os.path.join(losses_dir, f"{base_filename}_batch_losses.txt")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file_path))
    else:
        model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    print(f"Model loaded from: {model_file_path}")

    epoch_losses = []
    if os.path.exists(epoch_losses_path):
        with open(epoch_losses_path, 'r') as file:
            epoch_losses = [float(line.strip()) for line in file]
        print(f"Epoch losses loaded from: {epoch_losses_path}")

    batch_losses = []
    if os.path.exists(batch_losses_path):
        with open(batch_losses_path, 'r') as file:
            batch_losses = [float(line.strip()) for line in file]
        print(f"Batch losses loaded from: {batch_losses_path}")

    return model, epoch_losses, batch_losses


def parse_model_filename(filename):
    filename = filename.replace('.pth', '')

    # RE pattern for key-value pairs
    pattern = re.compile(r'([A-Za-z_]+)-([^_]+)')

    # Extract model type
    parts = filename.split('_', 1)
    model_type = parts[0]
    characteristics = {'Model': model_type}

    # Apply pattern to find key-value pairs
    matches = pattern.finditer(parts[1] if len(parts) > 1 else '')
    for match in matches:
        key = match.group(1).lstrip('_')  # Remove leading underscores from the key
        value = match.group(2)
        characteristics[key] = value

    return characteristics
