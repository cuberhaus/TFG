import os.path
import subprocess
import sys

from clases.cyclegan import prepareCycleGAN

POLYP_DATASET_DIR, _, cycleGan_dir, = prepareCycleGAN()
# test_dataset_dir = os.path.join(POLYP_DATASET_DIR, 'testA')
# print(test_dataset_dir)

python_installed = sys.executable
test_path = os.path.join(cycleGan_dir, 'test.py')

command = [
    python_installed,
    test_path,
    '--dataroot',
    POLYP_DATASET_DIR,
    '--name',
    'mask2polyp',
    '--model',
    'cycle_gan',
    '--no_dropout'
]
subprocess.run(command)
