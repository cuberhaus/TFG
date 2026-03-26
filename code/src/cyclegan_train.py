import os
import subprocess
import sys

from clases.cyclegan import prepareCycleGAN

batch_size = '4'
n_epochs = '5'

POLYP_DATASET_DIR, _, cycleGan_dir = prepareCycleGAN()

train_path = os.path.join(cycleGan_dir, 'train.py')

# We need to get the python executable path because subprocess doesn't use the same one as anaconda
python_installed = sys.executable
print(sys.executable)

# Construct the model name to include batch_size and n_epochs
model_name = f'mask2polyp_bs-{batch_size}_epochs-{n_epochs}'

# Constructing the command
command = [
    python_installed,
    train_path,
    '--dataroot',
    POLYP_DATASET_DIR,
    '--name',
    model_name,
    '--model',
    'cycle_gan',
    '--batch_size',
    batch_size,
    '--epoch_count',
    '1',
    '--n_epochs',
    n_epochs
]
print(command)
subprocess.run(command)

# command = f'"{python_installed}" "{train_path}" --dataroot "{POLYP_DATASET_DIR}" --name mask2polyp --model cycle_gan --display_id -1'
# os.system(command)
