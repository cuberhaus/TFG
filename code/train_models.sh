#!/bin/bash

# Check if a JSON file path is provided as an argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 path_to_json_file"
    exit 1
fi

# Assign the first argument to JSON_FILE variable
JSON_FILE="$1"

# Check if the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at $JSON_FILE"
    exit 1
fi

# Check if the second argument is --debug
DEBUG_FLAG=""
if [ "$#" -eq 2 ] && [ "$2" = "--debug" ]; then
    DEBUG_FLAG="--debug"
fi

# Use Python to parse the JSON file and loop through each entry
python3 -c "
import json, sys
with open('$JSON_FILE') as f:
    data = json.load(f)
    for entry in data:
        print(json.dumps(entry))
" | while read -r row; do
    model_name=$(echo $row | python3 -c "import json, sys; print(json.loads(sys.stdin.read())['model_name'])")
    params=$(echo $row | python3 -c "import json, sys; print(json.dumps(json.loads(sys.stdin.read())['params']))")

    # Print the model and its parameters
    echo "Running model: $model_name"
    echo "Parameters: $params"

    # Call the Python script with the model name and parameters
     python3 train_and_save_model.py "$model_name" "$params" $DEBUG_FLAG
done
