import os
import subprocess

from clases.cyclegan import prepareSPADE

POLYP_DATASET_DIR, python_installed, cycleGan_dir = prepareSPADE()
testA_dir = os.path.join(POLYP_DATASET_DIR, "testA")  # masks
testB_dir = os.path.join(POLYP_DATASET_DIR, "testB")  # images

train_dir = os.path.join(cycleGan_dir, "train.py")

command = (["python", train_dir, "--name", "spade_train", "--dataset_mode", "custom", "--label_dir", testA_dir,
            "--image_dir", testB_dir, "--label_nc", "2"])
print(command)

# To train on your own custom dataset python train.py --name [experiment_name] --dataset_mode custom --label_dir [
# path_to_labels] -- image_dir [path_to_images] --label_nc [num_labels]

subprocess.run(command)