import os
import subprocess

from clases.cyclegan import prepareSPADE

POLYP_DATASET_DIR, python_installed, cycleGan_dir = prepareSPADE()
A_dir = os.path.join(POLYP_DATASET_DIR, "trainA")  # masks
B_dir = os.path.join(POLYP_DATASET_DIR, "trainB")  # images

train_dir = os.path.join(cycleGan_dir, "train.py")

command = ([python_installed, train_dir, "--name", "spade_train", "--dataset_mode", "custom", "--label_dir", A_dir,
            "--image_dir", B_dir, "--label_nc", "2", "--no_instance"])
print(command)

# To train on your own custom dataset python train.py --name [experiment_name] --dataset_mode custom --label_dir [
# path_to_labels] -- image_dir [path_to_images] --label_nc [num_labels]

subprocess.run(command)