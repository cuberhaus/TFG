import json
import subprocess
import sys

# Replace '/path/to/src/clases' with the actual absolute path to the 'clases' directory
# sys.path.append('./src')

"""
This script is used to run the train_and_save_model.py script for each model in the JSON file.
"""


def main():
    # Check if a JSON file path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python script.py path_to_json_file [--debug]")
        sys.exit(1)

    # Assign the first argument to JSON_FILE variable
    json_file = sys.argv[1]

    # Check if the JSON file exists
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file}")
        sys.exit(1)

    # Check if the second argument is --debug
    debug_flag = ""
    if len(sys.argv) == 3 and sys.argv[2] == "--debug":
        debug_flag = "--debug"

    # Loop through each entry in the JSON file
    for entry in data:
        model_name = entry['model_name']
        params = json.dumps(entry['params'])

        # Print the model and its parameters
        print(f"Running model: {model_name}")
        print(f"Parameters: {params}")

        # Call the Python script with the model name and parameters
        subprocess.run(["python3", "src/train_and_save_model.py", model_name, params, debug_flag])


if __name__ == "__main__":
    main()
