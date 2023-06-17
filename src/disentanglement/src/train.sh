#!/bin/bash

# Check if the file containing the number exists
if [ ! -f counter.txt ]; then
    # If the file doesn't exist, create it and set the initial value to 1
    echo "1" > counter.txt
fi

# Read the current value from the file
current_number=$(cat counter.txt)

# Increment the number
next_number=$((current_number + 1))

# # Print the incremented number
# echo "Current number: $current_number"
# echo "Next number: $next_number"

# Update the file with the new number
echo "$next_number" > counter.txt

mkdir -p builds/"$current_number"/models

# "dataset_type" – s(f), s(f+cf), s(f+a), and s(f+cf+a)
# "-t" - lora/finetuning
# "-b" - batch_size
# "-m" – model_type large/xl/xxl 

python -u main_class.py --dataset_type 's(f)' -t lora -b 64 --grad_accum 1 -m large -s .builds/$current_number/models/ -g 80g -p $current_number &> $current_number.txt 