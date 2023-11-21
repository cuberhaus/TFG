#!/bin/bash

# Define remote host and directory
REMOTE_HOST="cuberhaus@teegarden.cs.upc.edu"
REMOTE_DIR="/home/cuberhaus/code/"

# Define an array of local file paths to be transferred
FILES=(
    "./custom_dataset.py"
    "./model_utils.py"
    "./train_and_save_model.py"
)

# Loop through each file and transfer it
for file in "${FILES[@]}"; do
    echo "Transferring $file to $REMOTE_HOST:$REMOTE_DIR"
    scp "$file" "$REMOTE_HOST":"$REMOTE_DIR"
done

echo "All files transferred to $REMOTE_HOST:$REMOTE_DIR"
