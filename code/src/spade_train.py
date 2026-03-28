import argparse
import os
import subprocess
import sys
import shutil

from clases.cyclegan import prepareSPADE


def _find_python() -> str:
    if os.path.basename(sys.executable).startswith("python"):
        return sys.executable
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            return path
    return "python3"


def main():
    parser = argparse.ArgumentParser(description="Train SPADE")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--niter", type=int, default=50, help="Epochs at initial LR")
    parser.add_argument("--niter-decay", type=int, default=0, help="Epochs to linearly decay LR to zero")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--netG", type=str, default="spade", choices=["spade", "pix2pixhd"])
    parser.add_argument("--load-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--max-dataset-size", type=int, default=None,
                        help="Limit samples per dataset folder (for quick testing)")
    args = parser.parse_args()

    POLYP_DATASET_DIR, _, spade_dir = prepareSPADE()
    A_dir = os.path.join(POLYP_DATASET_DIR, "trainA")
    B_dir = os.path.join(POLYP_DATASET_DIR, "trainB")
    train_path = os.path.join(spade_dir, "train.py")
    python_installed = _find_python()

    command = [
        python_installed, train_path,
        "--name", "spade_train",
        "--dataset_mode", "custom",
        "--label_dir", A_dir,
        "--image_dir", B_dir,
        "--label_nc", "1",
        "--no_instance",
        "--batchSize", str(args.batch_size),
        "--niter", str(args.niter),
        "--niter_decay", str(args.niter_decay),
        "--lr", str(args.lr),
        "--netG", args.netG,
        "--load_size", str(args.load_size),
        "--crop_size", str(args.crop_size),
    ]

    if args.max_dataset_size is not None:
        command.extend(["--max_dataset_size", str(args.max_dataset_size)])

    print(command)
    subprocess.run(command)


if __name__ == "__main__":
    main()