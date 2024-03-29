import utils
import torch
import argparse
import pandas as pd
import common_utils

from t5_finetuning_class import Finetuning
from t5_promptuning_class import PromptTuning
from t5_lightweight_tuning_class import LightweightTuning
from t5_adversarial_training_class import AdversarialTraining
from t5_inctxlearning_class import InCtxLearning

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():

    tuning_choices = [
        (Finetuning.get_aliases(), Finetuning),
        (PromptTuning.get_aliases(), PromptTuning),
        (InCtxLearning.get_aliases(), InCtxLearning),
        (LightweightTuning.get_aliases(), LightweightTuning),
        (AdversarialTraining.get_aliases(), AdversarialTraining),
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
    parser.add_argument( '-t', '--tuning', type=str, nargs='?',
                        default='finetuning', choices=all_tuning_choises)
    parser.add_argument( '--checkpoint_ids', nargs='+', required=True )
    parser.add_argument('--val_loss', default=False, action="store_true",
                        help='if True then val_acc otherwise val_loss')
    parser.add_argument('--skip_train', action="store_true", default=False, help='for testing purpose')

    # Parse the arguments
    args = parser.parse_args()

    print("\n============================")
    print(f'ARGUMENTS: {args}')
    print("============================\n")

    for checkpoint_id in args.checkpoint_ids:
        checkpoint_id = int(checkpoint_id)
        args.tuning = utils.get_tuning_type(checkpoint_id)
        print(f'{args.tuning=}')

        checkpoint = utils.find_best_checkpoint(checkpoint_id)
        with open(checkpoint+'/results.txt', 'r') as f:
            print(f.readlines())

        config = TrainingConfig(model_name=utils.get_model_name(args.model_name),
                                dataset_type=args.dataset_type, batch_size=args.batch_size,
                                num_gpus=1, gradient_accumulation_steps=1,
                                gpu_stat_every=500, 
                                evaluation_every=1, 
                                device=utils.deduce_device(), 
                                epochs=100, 
                                model_saving_folder='',
                                tuning_method=args.tuning,
                                gpu_name=args.gpu_name,
                                skip_train=args.skip_train,
                                val_accuracy=(not args.val_loss)
                    )

        print(vars(config))
        
        for choice in tuning_choices:
            if args.tuning not in choice[0]:
                continue

            method = choice[1](config, checkpoint)
            method.eval()

            break

if __name__ == '__main__':
    main()
