#!/bin/bash

# Assuming the JSON file is named 'model_configs.json'
JSON_FILE="model_configs.json"

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