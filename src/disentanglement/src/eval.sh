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

# "dataset_type" – s(f), s(f+cf), s(f+a), s(f+cf+a), ...
# "checkpoint_id" model checkpoint that you want to use
# "-t" - lora/finetuning
# "-b" - batch_size
# "-m" – model_type large/xl/xxl 

python -u t5_eval_class.py -m large --dataset_type 'gpt_rnd_2_f_jb' -b 32 -t finetuning --checkpoint_id 944 -s .builds/$current_number/models/ -g tesla -p $current_number &> $current_number.txt
