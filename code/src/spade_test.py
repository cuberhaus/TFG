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
    parser = argparse.ArgumentParser(description="Test SPADE (generate images)")
    parser.add_argument("--name", type=str, default="spade_train",
                        help="Name of the experiment")
    parser.add_argument("--which-epoch", type=str, default="latest",
                        help="Which epoch to load")
    args = parser.parse_args()

    POLYP_DATASET_DIR, _, spade_dir = prepareSPADE()
    test_path = os.path.join(spade_dir, "test.py")
    python_installed = _find_python()

    A_dir = os.path.join(POLYP_DATASET_DIR, "testA")
    B_dir = os.path.join(POLYP_DATASET_DIR, "testB")

    command = [
        python_installed, test_path,
        "--name", args.name,
        "--dataset_mode", "custom",
        "--label_dir", A_dir,
        "--image_dir", B_dir,
        "--label_nc", "1",
        "--no_instance",
        "--which_epoch", args.which_epoch,
    ]

    print(command)
    subprocess.run(command)


if __name__ == "__main__":
    main()
