#!/bin/bash

# Define remote host and base directory
REMOTE_HOST="casacuberta@teegarden.cs.upc.edu"
REMOTE_DIR="/home/casacuberta/TFG/code/"

# Define the local directories to be transferred
LOCAL_DIRS=("./src/" "./configurations/")

# Loop over each local directory and transfer it
for DIR in "${LOCAL_DIRS[@]}"; do
    echo "Transferring $DIR to $REMOTE_HOST:$REMOTE_DIR"
    scp -r "$DIR" "$REMOTE_HOST":"$REMOTE_DIR"
    echo "All files in $DIR transferred to $REMOTE_HOST:$REMOTE_DIR"
done