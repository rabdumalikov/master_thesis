import os
import wandb
import argparse
import utils
import numpy as np

from t5_finetuning_class import Finetuning
from t5_promptuning_class import PromptTuning
from t5_lightweight_tuning_class import LightweightTuning
from t5_adversarial_training_class import AdversarialTraining
from t5_inctxlearning_class import InCtxLearning

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_short_model_name(name):
    #choices = ['large', 'xl', 'xxl']

    short_name = name.upper() if 'x' in name else name.upper()[0]

    return f'T5{short_name}'

def main():
    model_name_choices = utils.get_model_name_choices()
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

    dataset_types_choices = utils.get_dataset_name_choices()

    # Create the parser
    parser = argparse.ArgumentParser(description='MasterThesis')

    # Add the positional argument
    parser.add_argument('-m', '--model_name', nargs='?',
                        default='large', choices=model_name_choices)
    parser.add_argument('-e', '--epochs', type=int, nargs='?',
                        default=100, help='number of epochs')
    parser.add_argument('--dataset_type', type=str,
                        help='type of dataset to train against', choices=dataset_types_choices)
    parser.add_argument('--fp16', type=bool, nargs='?',
                        default=True, help='enable or disable fp16')
    parser.add_argument('-p', '--process_id', type=int,
                        help='id of the process')
    parser.add_argument('-s', '--save_in', type=str,
                        help='folder for model saving')
    parser.add_argument('-g', '--gpu_name', type=str,
                        help='name of the gpu that will be used')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='size of the mini batch ')

    parser.add_argument('--grad_accum', type=int,
                        help='gradient accumulation steps')

    parser.add_argument(
        '-t', '--tuning', type=str, choices=all_tuning_choises)  # ['finetuning', 'prefixtuning', 'promptuning', 'adapters', 'lora', 'adversarial_training', 'in-context-learning'])

    # Parse the arguments
    args = parser.parse_args()

    config = utils.TrainingConfig(
        model_name=utils.get_model_name(args.model_name),
        gradient_accumulation_steps=args.grad_accum,
        batch_size=args.batch_size,
        gpu_stat_every=500, evaluation_every=1, num_gpus=utils.get_number_of_gpus(),
        device=utils.deduce_device(), dataset_type=args.dataset_type,
        epochs=utils.get_number_of_epochs(args.epochs),
        model_saving_folder=args.save_in,
        FP16=args.fp16,
        tuning_method=args.tuning
    )

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MasterThesis",
        id=str(args.process_id),
        group='HyperParametersTuning',
        name=f'id[{args.process_id}]-[{args.tuning}]-on[{args.dataset_type}]-bs[{config.batch_size}]-{utils.get_model_name(args.model_name)}',
        # track hyperparameters and run metadata
        config=vars(config)
    )

    print("\n============================")
    print(f'ARGUMENTS: {args}')
    print("============================\n")

    for choice in tuning_choices:
        if args.tuning in choice[0]:
            method = choice[1](config)
            best_em_score = method.train()

            print("\n============================\n")
            print(f'{best_em_score=}')

            wandb.log({'best_em_score': best_em_score})
            break

    wandb.finish()


if __name__ == "__main__":
    main()
