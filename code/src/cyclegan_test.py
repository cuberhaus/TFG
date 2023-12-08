import os.path
import subprocess

from clases.cyclegan import prepareCycleGAN

python_installed, POLYP_DATASET_DIR = prepareCycleGAN()
test_dataset_dir = os.path.join(POLYP_DATASET_DIR, 'testA')
print(test_dataset_dir)
subprocess.run(
    [python_installed, 'test.py', '--dataroot', test_dataset_dir, '--name', 'mask2polyp', '--model', 'test', '--no_dropout'])
