import os
import platform

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from model_utils import load_model_with_hyperparams, prepare_dataset, get_model


# Function to perform predictions and save images
def test_model_and_save_images(model, test_dataset, save_dir='testing_model'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for idx, (image, _) in enumerate(test_dataset):
            image = image.to(device).unsqueeze(0)  # Add batch dimension
            output = model(image)

            # Scale the tensor values to [0, 255] and convert to uint8
            scaled_image = image.squeeze(0).cpu().mul(255).byte()

            # Draw bounding boxes and labels on the image
            boxes = output[0]['boxes'].cpu()
            print(boxes)
            labels = output[0]['labels'].cpu()
            image_with_boxes = draw_bounding_boxes(scaled_image, boxes, labels=labels)

            # Convert to PIL Image for saving
            pil_image = to_pil_image(image_with_boxes)
            # Save the image
            pil_image.save(os.path.join(save_dir, f'output_{idx}.jpg'))


# Define model architecture
num_classes = 2  # 1 class + background
model = get_model('FasterRCNN', num_classes)

# Load the model weights
model_name = 'best_model.pth'  # Replace with your model's name
save_dir = './saved_models'  # Replace with the path to your saved models directory

# os.path.join(save_dir, '{model_name}.pth')
system_name = platform.system()

if system_name == 'Linux':
    print(save_dir)
    model, _, _ = load_model_with_hyperparams(model, model_name, load_dir=save_dir)  # Assuming the model is compatible with this function
elif system_name == 'Darwin':
    model, _, _ = load_model_with_hyperparams(model, model_name)  # Assuming the model is compatible with this function

debug = False
if debug == True:
    max_samples = {
        'train': 20,
        'test': 10
    }
else:
    max_samples = None

# Load the test dataset
train_dataset, test_dataset = prepare_dataset(max_samples)

# Test the model and save images
test_model_and_save_images(model, test_dataset)
