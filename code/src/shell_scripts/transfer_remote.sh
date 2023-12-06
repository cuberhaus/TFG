#!/bin/bash

# Define remote host and directory
REMOTE_HOST="casacuberta@teegarden.cs.upc.edu"
REMOTE_DIR="/home/casacuberta/TFG/code/"

# Define the local directory to be transferred
LOCAL_DIR="./src/"

# Transfer the entire directory
echo "Transferring $LOCAL_DIR to $REMOTE_HOST:$REMOTE_DIR"
scp -r "$LOCAL_DIR" "$REMOTE_HOST":"$REMOTE_DIR"

echo "All files in $LOCAL_DIR transferred to $REMOTE_HOST:$REMOTE_DIR"
