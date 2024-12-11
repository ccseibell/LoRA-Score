#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="ViT_training.py"

# Define dataset to be used
DATASET_1="stanford-dogs"
DATASET_2="oxford-pet"
DATASET=$DATASET_2

# Loop through the desired combinations
for shuffle_proportion in 0.05 0.1 0.15 0.2 0.3 0.4 0.5; do
    
    echo "Running all classes with shuffle proportion: $shuffle_proportion"
    
    # Run the Python script with the generated classes
    python $PYTHON_SCRIPT --dataset $DATASET --use_lora --shuffle_label_ratio $shuffle_proportion
    
    # Add a small delay to avoid overlaps (optional)
    sleep 1
done
