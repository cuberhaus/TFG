import cv2
import numpy as np

from clases.custom_dataset import parse_annotation
from clases.model_utils import *

"""
This script is used to create masks from the bounding boxes in the annotations files.
"""


def create_mask_from_bounding_boxes(image_shape, bounding_boxes):
    """
    Create a mask from the bounding boxes in the annotations file.
    :param image_shape:
    :param bounding_boxes:
    :return:
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Assuming image_shape is in (H, W, C) format
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255  # Or 1, if you want the mask to be in binary [0, 1]
    return mask


def process_images_in_folder(images_folder, annotations_folder, mask_save_dir):
    """
    Process all images in a folder.
    :param images_folder:
    :param annotations_folder:
    :param mask_save_dir:
    :return:
    """
    image_filenames = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    os.makedirs(mask_save_dir, exist_ok=True)  # Ensure the save directory exists

    for image_filename in image_filenames:
        img_path = os.path.join(images_folder, image_filename)
        ann_path = os.path.join(annotations_folder, image_filename.replace('.jpg', '.txt'))

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Image {img_path} not found or unable to open.")
            continue

        if not os.path.isfile(ann_path):
            print(f"Warning: Annotation file {ann_path} not found.")
            continue

        bboxes = parse_annotation(ann_path)
        mask = create_mask_from_bounding_boxes(image.shape, bboxes)

        mask_filename = os.path.splitext(image_filename)[0] + "_mask.png"
        mask_save_path = os.path.join(mask_save_dir, mask_filename)

        cv2.imwrite(mask_save_path, mask)
        print(f"Saved mask to {mask_save_path}")


def process_all_videos(root_dir):
    """
    Process all videos in the dataset.
    :param root_dir:
    :return:
    """
    images_root_dir = os.path.join(root_dir, 'Images')
    annotations_root_dir = os.path.join(root_dir, 'Annotations')
    mask_save_dir = os.path.join(root_dir, 'Masks')

    for video_number in sorted(os.listdir(images_root_dir)):
        video_images_folder = os.path.join(images_root_dir, video_number)
        video_annotations_folder = os.path.join(annotations_root_dir, video_number)
        video_mask_save_dir = os.path.join(mask_save_dir, video_number)

        if os.path.isdir(video_images_folder):
            process_images_in_folder(video_images_folder, video_annotations_folder, video_mask_save_dir)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example usage
dataset_root_dir = '../../data/TrainValid/TrainValid'  # Replace with the path to your TrainValid directory
dataset_root_dir = os.path.join(SCRIPT_DIR, dataset_root_dir)
process_all_videos(dataset_root_dir)

dataset_root_dir = '../../data/Test/Test'  # Replace with the path to your TrainValid directory
dataset_root_dir = os.path.join(SCRIPT_DIR, dataset_root_dir)
process_all_videos(dataset_root_dir)
