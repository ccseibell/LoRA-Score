#!/bin/bash

# Running script for all classes
python script.py --dataset oxford-pet

# Define the total number of classes in the dataset
TOTAL_CLASSES=37

# Function to generate a random combination of classes
generate_random_classes() {
    local num_classes=$1
    shuf -i 0-$(($TOTAL_CLASSES - 1)) -n $num_classes | tr '\n' ',' | sed 's/,$//'
}

# Define the Python script to run
PYTHON_SCRIPT="ViT_training.py"

# Loop through the desired combinations
for i in {1..10}; do
    for num_classes in 2 5 10 20; do
        # Generate random classes
        random_classes=$(generate_random_classes $num_classes)
        
        echo "Running for $num_classes random classes: $random_classes"
        
        # Run the Python script with the generated classes
        python $PYTHON_SCRIPT --num_classes $num_classes --class_ids $random_classes
        
        # Add a small delay to avoid overlaps (optional)
        sleep 1
    done
done
