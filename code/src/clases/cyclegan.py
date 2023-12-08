import os
import subprocess

from clases.utils import python_version


def prepareCycleGAN():
    # Script dir
    clases_dir = os.path.dirname(__file__)  # Directory of the script file
    src_dir = os.path.join(clases_dir, "../")  # Full path to the file
    cycleGan_dir = os.path.join(src_dir, "../tmp/pytorch-CycleGAN-and-pix2pix")  # Full path to the file
    # check if the directory exists
    if not os.path.exists(cycleGan_dir):
        subprocess.run(['git', 'clone', 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix', cycleGan_dir])
        requirements_dir = os.path.join(cycleGan_dir, 'requirements.txt')
        subprocess.run(['pip', 'install', '-r', requirements_dir])
    os.chdir(cycleGan_dir)
    POLYP_DATASET_DIR = os.path.join(src_dir, "../data/PolypDataset")
    # PYTHON_INSTALLED = is_python_installed()
    python_installed = python_version()
    return POLYP_DATASET_DIR, python_installed
