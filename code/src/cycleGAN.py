import os
import subprocess

# Script dir
script_dir = os.path.dirname(__file__)  # Directory of the script file
cycleGan_dir = os.path.join(script_dir, "../tmp/pytorch-CycleGAN-and-pix2pix")  # Full path to the file
# check if the directory exists
if not os.path.exists(cycleGan_dir):
    subprocess.run(['git', 'clone', 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix', cycleGan_dir])
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
os.chdir(cycleGan_dir)
