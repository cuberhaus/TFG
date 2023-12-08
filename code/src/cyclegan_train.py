import subprocess

from clases.cyclegan import prepareCycleGAN

python_installed, POLYP_DATASET_DIR = prepareCycleGAN()

subprocess.run(
    [python_installed, 'train.py', '--dataroot', POLYP_DATASET_DIR, '--name', 'mask2polyp', '--model', 'cycle_gan',
     '--display_id', '-1'])
