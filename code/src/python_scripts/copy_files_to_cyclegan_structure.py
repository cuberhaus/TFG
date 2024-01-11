import os
import shutil
from pathlib import Path

"""
This script copies the files from the PolypDetection dataset to the CycleGAN structure.
"""


# GAN typical training structure
# ├── datasets
#     └── your_dataset_name
#         ├── trainA  # Masks for training
#         ├── trainB  # Images for training
#         ├── testA   # Masks for testing
#         └── testB   # Images for testing

def copy_files_to_cyclegan_structure(root_dir, target_dir, mode='train'):
    assert mode in ('train', 'test'), "Mode must be 'train' or 'test'."

    images_root_dir = Path(root_dir) / 'Images'
    masks_root_dir = Path(root_dir) / 'Masks'

    trainA_dir = Path(target_dir) / f'{mode}A'
    trainB_dir = Path(target_dir) / f'{mode}B'

    trainA_dir.mkdir(parents=True, exist_ok=True)
    trainB_dir.mkdir(parents=True, exist_ok=True)

    print("Copying files to CycleGAN structure...")
    for video_number in sorted(images_root_dir.iterdir()):
        print(f"Processing video {video_number}")
        if video_number.is_dir():
            mask_subdir = masks_root_dir / video_number.name
            if not mask_subdir.exists() or not mask_subdir.is_dir():
                print(f"Mask directory does not exist: {mask_subdir}")
                continue

            image_subdir = images_root_dir / video_number.name
            if not image_subdir.exists() or not image_subdir.is_dir():
                print(f"Image directory does not exist: {image_subdir}")
                continue

            mask_files = list(mask_subdir.glob('*.png'))
            if not mask_files:
                print(f"No mask files found in directory: {mask_subdir}")
                continue

            for mask_file in mask_files:
                new_mask_name = mask_file.name.replace('_mask', '')
                new_mask_filename = f"{video_number.name}_{new_mask_name}"
                shutil.copy(mask_file, trainA_dir / new_mask_filename)

            image_files = list(image_subdir.glob('*.jpg'))
            if not image_files:
                print(f"No image files found in directory: {image_subdir}")
                continue

            for image_file in image_files:
                new_image_filename = f"{video_number.name}_{image_file.name}"
                shutil.copy(image_file, trainB_dir / new_image_filename)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# root_dir = os.path.join(SCRIPT_DIR, '../../data/TrainValid/TrainValid/')
# target_dir = os.path.join(SCRIPT_DIR, '../../data/PolypDatasetSPADE/')
# copy_files_to_cyclegan_structure(root_dir, target_dir, mode='train')

# root_dir = os.path.join(SCRIPT_DIR, '../../data/Test/Test/')
# target_dir = os.path.join(SCRIPT_DIR, '../../data/PolypDatasetSPADE/')
# copy_files_to_cyclegan_structure(root_dir, target_dir, mode='test')
