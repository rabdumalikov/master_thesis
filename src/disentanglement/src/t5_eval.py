import utils
import torch
import argparse
import pandas as pd
import common_utils
import t5_softprompt
import t5_finetuning
import t5_promptuning2
import t5_inctxlearning
import t5_adversarial_training

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

def find_best_checkpoint(id: int):
    dir_path = f'.builds/{id}/models'

    checkpoint_name = ''
    best_em = -1
    # iterate over all the directories in the specified directory
    for dir_name in os.listdir(dir_path):

        if os.path.isdir(os.path.join(dir_path, dir_name)):

            with open(dir_path + '/' + dir_name  + '/' + 'results.txt', 'r') as f:
                d = {}
                for l in f.readlines():
                    key, value = l.split('=')
                    d[key] = value
                    

                if float(d['em']) > best_em:
                    best_em = float(d['em'])
                    checkpoint_name = dir_name

                    print(f'Best checkpoint {checkpoint_name} with em={best_em}')

    return dir_path + '/' + checkpoint_name


def main():

    tuning_choices = [
        (t5_softprompt.get_aliases(), t5_softprompt.create_stuff),
        (t5_finetuning.get_aliases(), t5_finetuning.create_stuff),
        (t5_promptuning2.get_aliases(), t5_promptuning2.create_stuff),
        (t5_inctxlearning.get_aliases(), t5_inctxlearning.create_stuff),
        (t5_adversarial_training.get_aliases(), t5_adversarial_training.create_stuff),
    ]

    all_tuning_choises = []
    for c in tuning_choices:
        all_tuning_choises.extend(c[0])

    # Create the parser
    parser = argparse.ArgumentParser(description='MasterThesis')

    # Add the positional argument
    parser.add_argument('-m', '--model_name', nargs='?',
                        default='large', choices=utils.get_model_name_choices())
    parser.add_argument('--dataset_type', type=str,
                        help='type of dataset to train against', choices=utils.get_dataset_name_choices())
    parser.add_argument('-p', '--process_id', type=int,
                        help='id of the process')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='size of the mini batch ')
    parser.add_argument('-s', '--save_in', type=str,
                        help='folder for model saving')
    parser.add_argument('-g', '--gpu_name', type=str,
                        help='name of the gpu that will be used')
    parser.add_argument( '-t', '--tuning', type=str, choices=all_tuning_choises)
    parser.add_argument( '--checkpoint_id', type=int )

    # Parse the arguments
    args = parser.parse_args()


    checkpoint = find_best_checkpoint(args.checkpoint_id) #275
    with open(checkpoint+'/results.txt', 'r') as f:
        print(f.readlines())

    config = TrainingConfig(model_name=utils.get_model_name(args.model_name),
                            dataset_type=args.dataset_type, batch_size=args.batch_size,
                            num_gpus=1, gradient_accumulation_steps=1,
                            gpu_stat_every=500, 
                            evaluation_every=1, 
                            device=deduce_device(), 
                            epochs=100, 
                            model_saving_folder='',
                            tuning_method=args.tuning
                )

    print(config)
    
    for choice in tuning_choices:
        if args.tuning not in choice[0]:
            continue

        training_elems, training_data = choice[1](config, checkpoint)
        torch.cuda.empty_cache()

        for k in training_data.test_loaders:
            acc = evaluate(training_elems, config, training_data.test_loaders[k], verbose=True)
            print(f'{k=} {acc=}')

        break

if __name__ == '__main__':
    main()
