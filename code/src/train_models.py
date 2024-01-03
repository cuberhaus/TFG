import json
import os
import subprocess
import sys

# Replace '/path/to/src/clases' with the actual absolute path to the 'clases' directory
# sys.path.append('./src')

"""
This script is used to run the train_and_save_model.py script for each model in the JSON file.
"""

# This line gets the directory in which the script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_models.py path_to_json_file [--debug]")
        sys.exit(1)

    json_file = sys.argv[1]

    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file}")
        sys.exit(1)

    debug_flag = ""
    if len(sys.argv) == 3 and sys.argv[2] == "--debug":
        debug_flag = "--debug"

    for entry in data:
        model_name = entry['model_name']
        params = json.dumps(entry['params'])

        print(f"Running model: {model_name}")
        print(f"Parameters: {params}")

        train_and_save_model_script = os.path.join(SCRIPT_DIR, "train_and_save_model.py")

        subprocess.run(["python3", train_and_save_model_script, model_name, params, debug_flag])


if __name__ == "__main__":
    main()
