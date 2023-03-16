import wandb
import argparse 
import utils
import t5_ptuning

def main():

    # Create the parser
    parser = argparse.ArgumentParser(description='MasterThesis')

    # Add the positional argument
    parser.add_argument('-m', '--model_name', nargs='?', default='large', choices=['large', 'xl', 'xxl'])
    parser.add_argument('-e', '--epoch', type=int, nargs='?', default=100, help='number of epochs')
    parser.add_argument('--exp_id', type=int, help='experiment id')
    parser.add_argument('-p', '--process_id', type=int, help='id of the process')
    parser.add_argument('-f', '--folder', type=str, help='folder for output')
    parser.add_argument('-g', '--gpu_name', type=str,
                        help='name of the gpu that will be used')

    parser.add_argument('-t', '--tuning', choices=['ftuning', 'ptuning', 'adapters', 'lora', 'adversarial_training'])

    # Parse the arguments
    args = parser.parse_args()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MasterThesis",
        id=str(args.process_id),
        name=f'MT-T5_{args.model_name}-{args.tuning}-{args.process_id}',   
        # track hyperparameters and run metadata
        config=vars(training_config)
    )

    training_config = TrainingConfig(
        model_name=utils.get_model_name(args.model_name),
        gradient_accumulation_steps=2 if args.gpu_name == '40g' else 1,
        batch_size=16 if args.gpu_name == '40g' else 32,
        # math.ceil(50000 / (len(train_set)//32))
        gpu_stat_every=500, evaluation_every=1, num_gpus=get_number_of_gpus(),
        device=utils.deduce_device(), experiment_id=args.exp_id, epochs=100
    )


    if args.tuning == 'ftuning':

    elif args.tuning == 'ptuning':
        t5_ptuning.run( training_config )

    wandb.finish()



if __name__ == "__main__":
    main()
