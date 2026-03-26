import os.path
import subprocess
import sys
import argparse

from clases.cyclegan import prepareCycleGAN

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='mask2polyp', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
args = parser.parse_args()

POLYP_DATASET_DIR, _, cycleGan_dir = prepareCycleGAN()

python_installed = sys.executable
test_path = os.path.join(cycleGan_dir, 'test.py')

command = [
    python_installed,
    test_path,
    '--dataroot',
    POLYP_DATASET_DIR,
    '--name',
    args.name,
    '--model',
    'cycle_gan',
    '--no_dropout',
    '--epoch',
    args.epoch
]
subprocess.run(command)
