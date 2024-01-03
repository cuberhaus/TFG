import os
import platform

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from clases.model_utils import load_model_with_hyperparams, prepare_dataset, get_model

"""
This script performs predictions and saves the resulting images
"""


def test_model_and_save_images(model, test_dataset, class_labels, save_dir='out/testing_model', debug=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for idx, (image, targets) in enumerate(test_dataset):
            if not debug:
                print(f"Processing image {idx}")
            image = image.to(device).unsqueeze(0)  # Add batch dimension
            output = model(image)

            # Scale to [0, 255] and convert it to uint8
            scaled_image = image.squeeze(0).cpu().mul(255).byte()

            gt_boxes = targets['boxes'].cpu()
            image_with_gt_boxes = draw_bounding_boxes(scaled_image.clone(), gt_boxes, colors="green")

            boxes = output[0]['boxes'].cpu()
            if debug:
                print(boxes)
            label_indices = output[0]['labels'].cpu()

            string_labels = [class_labels[i.item()] for i in label_indices]

            image_with_boxes = draw_bounding_boxes(image_with_gt_boxes, boxes, labels=string_labels, colors="red")

            pil_image = to_pil_image(image_with_boxes)
            pil_image.save(os.path.join(save_dir, f'output_{idx}.jpg'))


model_name = 'best_model.pth'
save_dir = './saved_models'

num_classes = 2  # 1 class + background
model = get_model('FasterRCNN', num_classes)

system_name = platform.system()
if system_name == 'Linux':
    print(save_dir)
    model, _, _ = load_model_with_hyperparams(model, model_name,
                                              load_dir=save_dir)
elif system_name == 'Darwin':
    model, _, _ = load_model_with_hyperparams(model, model_name)

debug = False
if debug:
    max_samples = {
        'train': 20,
        'test': 10
    }
else:
    max_samples = None

train_dataset, test_dataset = prepare_dataset(max_samples)

class_labels = {0: 'background', 1: 'polyp'}
test_model_and_save_images(model, test_dataset, class_labels, debug=debug)
