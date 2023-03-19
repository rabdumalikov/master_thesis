import os
import wandb
import argparse
import utils

import t5_softprompt
import t5_finetuning
import t5_promptuning2
import t5_inctxlearning
import t5_adversarial_training

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():

    choises = [
        (t5_softprompt.get_aliases(), t5_softprompt.run),
        (t5_finetuning.get_aliases(), t5_finetuning.run),
        (t5_promptuning2.get_aliases(), t5_promptuning2.run), 
        (t5_inctxlearning.get_aliases(), t5_inctxlearning.run), 
        (t5_adversarial_training.get_aliases(), t5_adversarial_training.run), 
        ]

    all_choises = [ c[0] for c in choises ]

    # Create the parser
    parser = argparse.ArgumentParser(description='MasterThesis')

    # Add the positional argument
    parser.add_argument('-m', '--model_name', nargs='?',
                        default='large', choices=['large', 'xl', 'xxl'])
    parser.add_argument('-e', '--epochs', type=int, nargs='?',
                        default=100, help='number of epochs')
    parser.add_argument('--exp_id', type=int, help='experiment id')
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
        '-t', '--tuning', choices=all_choises) #['finetuning', 'prefixtuning', 'promptuning', 'adapters', 'lora', 'adversarial_training', 'in-context-learning'])

    # Parse the arguments
    args = parser.parse_args()

    config = utils.TrainingConfig(
        model_name=utils.get_model_name(args.model_name),
        gradient_accumulation_steps=args.grad_accum,
        batch_size=args.batch_size,
        gpu_stat_every=500, evaluation_every=1, num_gpus=utils.get_number_of_gpus(),
        device=utils.deduce_device(), experiment_id=args.exp_id,
        epochs=utils.get_number_of_epochs(args.epochs),
        model_saving_folder=args.save_in,
        FP16=args.fp16
    )

    experiment_name = get_experiment_alias(args.exp_id)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MasterThesis",
        id=str(args.process_id),
        group='Experiment_1',
        name=f'T5_{args.model_name}-[{args.tuning}]-on[{experiment_name}]-id[{args.process_id}]',
        # track hyperparameters and run metadata
        config=vars(config)
    )

    # T5_large-[finetuning]-on[s(f+cf+a)]-id[200]
    # T5_large-[prefix-tuning]-on[s(f)]-id[201]
    # T5_large-[bottleneck-adapters]-on[s(f+cf+a)]-id[203]
    # T5_large-[lora]-on[s(f+cf+a)]-id[204]
    # T5_large-[promptuning]-on[s(f+cf+a)]-id[205]
    # T5_large-[adversarial-training]-on[s(f+cf+a)]-id[206]

    print("\n============================")
    print(f'ARGUMENTS: {args}')
    print("============================\n")

    for choice in choices:
        if args.tuning in choice[0]:
            best_em_score = choice[1](config, args.tuning)

            print("\n============================\n")
            print(f'{best_em_score=}')

            wandb.log({'best_em_score': best_em_score})
            break

    wandb.finish()


if __name__ == "__main__":
    main()
