#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="ViT_training.py"
DATASET="oxford-pet"

# Running script for all classes
echo "Running all classes"
python $PYTHON_SCRIPT --dataset $DATASET

# Define the total number of classes in the dataset
TOTAL_CLASSES=37

# Function to generate a random combination of classes
generate_random_classes() {
    local num_classes=$1
    shuf -i 0-$(($TOTAL_CLASSES - 1)) -n $num_classes | tr '\n' ',' | sed 's/,$//'
}

# Loop through the desired combinations
for i in {1..10}; do
    for num_classes in 5 10 15 20 25; do
        # Generate random classes
        random_classes=$(generate_random_classes $num_classes)
        
        echo "Running for $num_classes random classes: $random_classes"
        
        # Run the Python script with the generated classes
        python $PYTHON_SCRIPT --dataset $DATASET --classes $random_classes
        
        # Add a small delay to avoid overlaps (optional)
        sleep 1
    done
done
