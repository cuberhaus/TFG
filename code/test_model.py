import os

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

            # Draw bounding boxes and labels on the image
            boxes = output[0]['boxes'].cpu()
            labels = output[0]['labels'].cpu()
            image_with_boxes = draw_bounding_boxes(to_pil_image(image.squeeze(0).cpu()), boxes, labels=labels)

            # Save the image
            image_with_boxes.save(os.path.join(save_dir, f'output_{idx}.jpg'))


# Define model architecture
num_classes = 2  # 1 class + background
model = get_model('FasterRCNN', num_classes)

# Load the model weights
model_name = 'best_model.pth'  # Replace with your model's name
model, _, _ = load_model_with_hyperparams(model, model_name, load_dir="./")  # Assuming the model is compatible with this function

max_samples = {
    'train': 20,
    'test': 10
}
# Load the test dataset
# test_root_dir = ''  # Replace with the path to your test dataset
train_dataset, test_dataset = prepare_dataset(max_samples)
# test_dataset = CustomDataset(root_dir=test_root_dir, transform=None)  # Assuming no additional transform needed

# Test the model and save images
test_model_and_save_images(model, test_dataset)
