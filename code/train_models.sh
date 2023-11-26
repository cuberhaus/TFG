#!/bin/bash

# Check if a JSON file path is provided as an argument
if [ "$#" -ne 1 ]; then
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

# Assuming the JSON file is named 'model_configs.json'
#JSON_FILE="model_configs.json"

# Read and loop through each entry in the JSON file
for row in $(cat $JSON_FILE | jq -c '.[]'); do
    model_name=$(echo $row | jq -r '.model_name')
    params=$(echo $row | jq -c '.params')

    # Print the model and its parameters
    echo "Running model: $model_name"
    echo "Parameters: $params"

    # Call the Python script with the model name and parameters
#    python your_script.py "$model_name" "$params"
done