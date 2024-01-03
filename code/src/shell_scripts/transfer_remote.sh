#!/bin/bash

REMOTE_HOST="casacuberta@teegarden.cs.upc.edu"
REMOTE_DIR="/home/casacuberta/TFG/code/"

LOCAL_DIRS=("./src/" "./configurations/")

for DIR in "${LOCAL_DIRS[@]}"; do
    echo "Transferring $DIR to $REMOTE_HOST:$REMOTE_DIR"
    scp -r "$DIR" "$REMOTE_HOST":"$REMOTE_DIR"
    echo "All files in $DIR transferred to $REMOTE_HOST:$REMOTE_DIR"
done