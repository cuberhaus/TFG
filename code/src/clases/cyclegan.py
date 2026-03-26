import os
import subprocess

from clases.utils import python_version


def prepareCycleGAN():
    clases_dir = os.path.dirname(__file__)
    src_dir = os.path.join(clases_dir, "../")
    cycleGan_dir = os.path.join(src_dir, "../tmp/pytorch-CycleGAN-and-pix2pix")

    if not os.path.exists(cycleGan_dir):
        subprocess.run(['git', 'clone', 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix', cycleGan_dir])
        requirements_dir = os.path.join(cycleGan_dir, 'requirements.txt')
        subprocess.run(['pip', 'install', '-r', requirements_dir])
    os.chdir(cycleGan_dir)
    POLYP_DATASET_DIR = os.path.join(src_dir, "../data/PolypDataset")
    python_installed = python_version()
    
    # Ensure dataset directories exist
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'trainA'), exist_ok=True)
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'trainB'), exist_ok=True)
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'testA'), exist_ok=True)
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'testB'), exist_ok=True)

    print(POLYP_DATASET_DIR)
    print(python_installed)
    return POLYP_DATASET_DIR, python_installed, cycleGan_dir

def prepareSPADE():
    clases_dir = os.path.dirname(__file__)
    src_dir = os.path.join(clases_dir, "../")
    cycleGan_dir = os.path.join(src_dir, "../tmp/SPADE")
    extra_dir = os.path.join(cycleGan_dir, "models/networks/Synchronized-BatchNorm-PyTorch")
    copy_from_dir = os.path.join(cycleGan_dir, "models/networks/Synchronized-BatchNorm-PyTorch/sync_batchnorm")
    copy_to_dir = os.path.join(cycleGan_dir, "models/networks/sync_batchnorm")
    print(copy_from_dir)
    print(copy_to_dir)

    if not os.path.exists(cycleGan_dir):
        subprocess.run(['git', 'clone', 'https://github.com/NVlabs/SPADE.git', cycleGan_dir])
        requirements_dir = os.path.join(cycleGan_dir, 'requirements.txt')
        subprocess.run(['pip', 'install', '-r', requirements_dir])
        subprocess.run(['git', 'clone', 'https://github.com/vacancy/Synchronized-BatchNorm-PyTorch', extra_dir])
    # subprocess.run(['cp', '-rf', 'Synchronized-BatchNorm-PyTorch/sync_batchnorm', extra_dir])
        subprocess.run(['cp', "-rf", copy_from_dir, copy_to_dir])  # in powershell -Recurse -Force


    # cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
    # cd ../../
    POLYP_DATASET_DIR = os.path.join(src_dir, "../data/PolypDatasetSPADE")
    python_installed = python_version()
    
    # Ensure dataset directories exist
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'trainA'), exist_ok=True)
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'trainB'), exist_ok=True)
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'testA'), exist_ok=True)
    os.makedirs(os.path.join(POLYP_DATASET_DIR, 'testB'), exist_ok=True)
    
    print(POLYP_DATASET_DIR)
    print(python_installed)
    return POLYP_DATASET_DIR, python_installed, cycleGan_dir
