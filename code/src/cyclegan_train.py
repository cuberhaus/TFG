import argparse
import os
import subprocess
import sys
import shutil

from clases.cyclegan import prepareCycleGAN


def _find_python() -> str:
    if os.path.basename(sys.executable).startswith("python"):
        return sys.executable
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            return path
    return "python3"


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--max-dataset-size", type=int, default=None,
                        help="Limit samples per dataset folder (for quick testing)")
    parser.add_argument("--netG", type=str, default="resnet_9blocks",
                        choices=["resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128"])
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()

    POLYP_DATASET_DIR, _, cycleGan_dir = prepareCycleGAN()
    train_path = os.path.join(cycleGan_dir, 'train.py')
    python_installed = _find_python()

    model_name = f'mask2polyp_bs-{args.batch_size}_epochs-{args.n_epochs}'

    command = [
        python_installed,
        train_path,
        '--dataroot', POLYP_DATASET_DIR,
        '--name', model_name,
        '--model', 'cycle_gan',
        '--batch_size', str(args.batch_size),
        '--epoch_count', '1',
        '--n_epochs', str(args.n_epochs),
        '--lr', str(args.lr),
        '--netG', args.netG,
        '--load_size', str(args.load_size),
        '--crop_size', str(args.crop_size),
        '--display_id', '-1',
    ]

    if args.max_dataset_size is not None:
        command.extend(['--max_dataset_size', str(args.max_dataset_size)])

    print(command)
    subprocess.run(command)


if __name__ == "__main__":
    main()
