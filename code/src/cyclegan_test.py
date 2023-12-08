import subprocess

from clases.cyclegan import prepareCycleGAN

python_installed, POLYP_DATASET_DIR = prepareCycleGAN()

subprocess.run(
    ['python3', 'test.py', '--dataroot', POLYP_DATASET_DIR, '--name', 'mask2polyp', '--model', 'test', '--no_dropout'])
