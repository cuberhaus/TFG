import subprocess
import os

"""
This script transfers all the files in the files array to the remote host
"""

# Define remote host and directory
remote_host = "casacuberta@teegarden.cs.upc.edu"
remote_dir = "/home/casacuberta/TFG/code/"

# Define an array of local file paths to be transferred
files = [
    "./custom_dataset.py",
    "./model_utils.py",
    "./optuna_train_model.py",
    "./raytune_train_model.py",
    "./train_and_save_model.py",

    "./python_scripts/test_model.py",
    "./shell_scripts/train_models.sh",
    "./configurations/model.json"
]

# Loop through each file and transfer it
for file in files:
    # Extract the directory path relative to the root directory
    relative_dir = os.path.dirname(file)
    remote_path = os.path.join(remote_dir, relative_dir)

    # Create the corresponding directory on the remote server
    if relative_dir:  # Check if the file is not in the root directory
        subprocess.run(["ssh", remote_host, f"mkdir -p {remote_path}"])

    # Extract the base filename
    basefile = os.path.basename(file)
    full_remote_path = os.path.join(remote_path, basefile)
    print(f"Transferring {file} to {remote_host}:{full_remote_path}")

    # Execute scp command to transfer the file
    scp_command = ["scp", file, f"{remote_host}:{full_remote_path}"]
    subprocess.run(scp_command)

print(f"All files transferred to {remote_host}:{remote_dir}")
