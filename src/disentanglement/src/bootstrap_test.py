import utils
import torch
import argparse
import numpy as np
import pandas as pd
import common_utils

from torch.utils.data import DataLoader

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
    parser.add_argument('-n', '--number_trials', type=int)
    parser.add_argument('-m', '--model_name', nargs='?',
                        default='large', choices=utils.get_model_name_choices())
    parser.add_argument('--dataset_type', type=str,
                        help='type of dataset to train against', choices=utils.get_dataset_name_choices())
    parser.add_argument('--testset_type', type=str,
                        help='type of testset to validate against', choices=['f', 'cf', 'a(e)', 'a(r)'])
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


    checkpoint = utils.find_best_checkpoint(args.checkpoint_id)
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
                            tuning_method=args.tuning,
                            gpu_name=args.gpu_name
                )

    print(config)
    
    for choice in tuning_choices:
        if args.tuning not in choice[0]:
            continue

        method = choice[1](config, checkpoint)

        for k in method.training_data.test_loaders:

            if k != 'cf':
                continue

            accs = []
            for i in range(args.number_trials):
                sample_test_set = method.training_data.test_loaders[k].dataset.get_dataframe().sample(n=1365, replace=True)

                stest_loader = DataLoader(utils.PandasDataset(sample_test_set), collate_fn=lambda inp: utils.collate_fn(
                    inp, max_source_input_len=396, tokenizer=method.training_elems.tokenizer), batch_size=config.eval_batch_size, shuffle=True, num_workers=4, pin_memory=True)

                acc = evaluate(method.training_elems, method.config, stest_loader, verbose=True)
                accs.append(acc)
                print(f'{i=} {acc=}')
            
            print(accs)
            print(np.mean(accs))
            print(np.std(accs))


        break

if __name__ == '__main__':
    main()
