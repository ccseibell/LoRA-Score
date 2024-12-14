#!/bin/bash

# Loop through the desired combinations
for shuffle_proportion in 0.05 0.1 0.15 0.2 0.3 0.4 0.5; do
    echo "Shuffling $shuffle_proportion of dataset"

    # Run the Python scripts
    python ViT_training.py --dataset oxford-pet --use_lora --shuffle_label_ratio $shuffle_proportion
    python ViT_training.py --dataset stanford-dogs --use_lora --shuffle_label_ratio $shuffle_proportion
    python BERT_training.py --dataset arxiv --use_lora --shuffle_label_ratio $shuffle_proportion
    
    # Add a small delay to avoid overlaps (optional)
    sleep 1
done
