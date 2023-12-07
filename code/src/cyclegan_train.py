import os
import subprocess
import clases.model_utils as model_utils

# Script dir
script_dir = os.path.dirname(__file__)  # Directory of the script file
cycleGan_dir = os.path.join(script_dir, "../tmp/pytorch-CycleGAN-and-pix2pix")  # Full path to the file
# check if the directory exists
if not os.path.exists(cycleGan_dir):
    subprocess.run(['git', 'clone', 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix', cycleGan_dir])
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
os.chdir(cycleGan_dir)

test_root_dir, train_root_dir = model_utils.dataset_paths()

subprocess.run(['python', 'train.py', '--dataroot', train_root_dir, '--name', 'mask2polyp', '--model', 'cycle_gan', '--display_id', '-1'])
