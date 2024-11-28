#!/bin/bash

# Ensure the script is executable
# chmod +x run_experiments.sh

# Define dataset sizes
train_sizes=(5000 10000 40000)
test_sizes=(1000 2000 10000)

# Loop over each combination of train and test sizes
for i in ${!train_sizes[@]}; do
    train_size=${train_sizes[$i]}
    test_size=${test_sizes[$i]}
    
    echo "Running experiment with train size: $train_size and test size: $test_size"
    
    # Run the Python script with the specified train and test sizes
    python analysis.py --train-size $train_size --test-size $test_size
    
    echo "Experiment completed for train size: $train_size and test size: $test_size"
    echo "-------------------------------------------------------------"
done