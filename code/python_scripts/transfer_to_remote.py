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
    "./test_model.py",
    "./optuna_train_model.py",
    "./raytune_train_model.py",
    "./train_and_save_model.py",
    "./train_models.sh",
    "./model.json"
]

# Loop through each file and transfer it
for file in files:
    # Extract the base filename
    basefile = os.path.basename(file)
    print(f"Transferring {file} to {remote_host}:{remote_dir}{basefile}")

    # Execute scp command
    scp_command = ["scp", file, f"{remote_host}:{remote_dir}{basefile}"]
    subprocess.run(scp_command)

print(f"All files transferred to {remote_host}:{remote_dir}")
