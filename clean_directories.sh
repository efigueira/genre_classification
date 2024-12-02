#!/bin/bash

# Define directories to search and remove
TARGETS=("mlruns" "outputs" "wandb" "artifacts")

echo "Starting cleanup..."

for target in "${TARGETS[@]}"; do
  echo "Searching for directories named '$target'..."
  find . -type d -name "$target*" -exec rm -rf {} + \
    && echo "Removed all '$target' directories." \
    || echo "No '$target' directories found or an error occurred."
done

echo "Cleanup complete!"
