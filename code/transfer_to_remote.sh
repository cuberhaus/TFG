#!/bin/bash

# Define remote host and directory
REMOTE_HOST="casacuberta@teegarden.cs.upc.edu"
REMOTE_DIR="/home/casacuberta/TFG/code/"

# Define an array of local file paths to be transferred
FILES=(
    "./custom_dataset.py"
    "./model_utils.py"
    "./train_and_save_model.py"
)

# Loop through each file and transfer it
for file in "${FILES[@]}"; do
    # Extract the base filename
    basefile=$(basename "$file")
    echo "Transferring $file to $REMOTE_HOST:$REMOTE_DIR$basefile"
    # shellcheck disable=SC2140
    scp "$file" "$REMOTE_HOST":"$REMOTE_DIR$basefile"
done

echo "All files transferred to $REMOTE_HOST:$REMOTE_DIR"
