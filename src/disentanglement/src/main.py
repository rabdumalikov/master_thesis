import os
import wandb
import argparse
import utils

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():

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
        '-t', '--tuning', choices=['ftuning', 'ptuning', 'promptuning', 'adapters', 'lora', 'adversarial_training', 'in-context-learning'])

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

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MasterThesis",
        id=str(args.process_id),
        name=f'T5_{args.model_name}-{args.tuning}-{args.process_id}',
        # track hyperparameters and run metadata
        config=vars(config)
    )

    if args.tuning == 'ftuning':
        import t5_ptuning
        best_em_score = t5_finetuning.run(config)
    elif args.tuning == 'ptuning':
        import t5_finetuning
        best_em_score = t5_ptuning.run(config)
    elif args.tuning == 'promptuning':
        import t5_promptuning
        best_em_score = t5_promptuning.run(config)
    elif args.tuning == 'in-context-learning':
        import t5_inctxlearning
        best_em_score = t5_inctxlearning.run(config)

    print("\n============================\n")
    print(f'{best_em_score=}')

    wandb.log({'best_em_score': best_em_score})

    wandb.finish()


if __name__ == "__main__":
    main()
