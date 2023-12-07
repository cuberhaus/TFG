import os

from clases.utils import *

# Script dir
script_dir = os.path.dirname(__file__)  # Directory of the script file
cycleGan_dir = os.path.join(script_dir, "../tmp/pytorch-CycleGAN-and-pix2pix")  # Full path to the file
# check if the directory exists
if not os.path.exists(cycleGan_dir):
    subprocess.run(['git', 'clone', 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix', cycleGan_dir])
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
os.chdir(cycleGan_dir)

POLYP_DATASET_DIR = os.path.join(script_dir, "../data/PolypDataset")

# PYTHON_INSTALLED = is_python_installed()
python_installed = python_version()

subprocess.run([python_installed, 'train.py', '--dataroot', POLYP_DATASET_DIR, '--name', 'mask2polyp', '--model', 'cycle_gan',
                '--display_id', '-1'])
