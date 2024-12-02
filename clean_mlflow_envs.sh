#!/bin/bash

echo "Searching for Conda environments starting with 'mlflow-'..."

# Get the list of environments that start with 'mlflow-'
MLFLOW_ENVS=$(conda env list | awk '{print $1}' | grep '^mlflow-')

if [[ -z "$MLFLOW_ENVS" ]]; then
  echo "No Conda environments found starting with 'mlflow-'."
  exit 0
fi

echo "Found the following environments:"
echo "$MLFLOW_ENVS"

# Loop through each environment and remove it
for env in $MLFLOW_ENVS; do
  echo "Removing environment: $env"
  conda remove -n "$env" --all -y
  if [[ $? -eq 0 ]]; then
    echo "Successfully removed: $env"
  else
    echo "Failed to remove: $env"
  fi
done

echo "Cleanup of 'mlflow-' environments complete!"
