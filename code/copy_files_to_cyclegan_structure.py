import shutil
from pathlib import Path


def copy_files_to_cyclegan_structure(root_dir, target_dir):
    # Define the source directories
    images_root_dir = Path(root_dir) / 'Images'
    masks_root_dir = Path(root_dir) / 'Masks'

    # Define the target directories
    trainA_dir = Path(target_dir) / 'trainA'  # Masks
    trainB_dir = Path(target_dir) / 'trainB'  # Original images

    # Create the target directories if they don't exist
    trainA_dir.mkdir(parents=True, exist_ok=True)
    trainB_dir.mkdir(parents=True, exist_ok=True)

    # Copy masks to trainA and images to trainB
    print("Copying files to CycleGAN structure...")
    for video_number in sorted(images_root_dir.iterdir()):
        print(f"Processing video {video_number}")
        if video_number.is_dir():  # Check if it is a directory
            mask_subdir = masks_root_dir / video_number.name
            # print(mask_subdir)
            if not mask_subdir.exists() or not mask_subdir.is_dir():
                print(f"Mask directory does not exist: {mask_subdir}")
                continue

            image_subdir = images_root_dir / video_number.name
            # print(image_subdir)
            if not image_subdir.exists() or not image_subdir.is_dir():
                print(f"Image directory does not exist: {image_subdir}")
                continue

            # Copy mask files to trainA
            mask_files = list(mask_subdir.glob('*.png'))
            if not mask_files:
                print(f"No mask files found in directory: {mask_subdir}")
                continue

            for mask_file in mask_files:
                new_mask_filename = f"{video_number.name}_{mask_file.name}"
                shutil.copy(mask_file, trainA_dir / new_mask_filename)
                # print(f"Copied {mask_file} to {trainA_dir / new_mask_filename}")

            # Copy image files to trainB
            image_files = list(image_subdir.glob('*.jpg'))
            if not image_files:
                print(f"No image files found in directory: {image_subdir}")
                continue

            for image_file in image_files:
                new_image_filename = f"{video_number.name}_{image_file.name}"
                shutil.copy(image_file, trainB_dir / new_image_filename)
                # print(f"Copied {image_file} to {trainB_dir / new_image_filename}")


# Example usage
root_dir = './data/TrainValid/TrainValid/'  # Replace with the path to your TrainValid directory
target_dir = './data/PolypDataset/'  # Replace with the path to your target dataset directory for CycleGAN
copy_files_to_cyclegan_structure(root_dir, target_dir)
