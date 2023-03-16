import os
import wandb
import torch
import argparse

from utils import *
from datetime import timedelta
from timeit import default_timer as timer
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, CompacterConfig, LoRAConfig, IA3Config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def create_tokenizer(model_name: str) -> T5Tokenizer:
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    SPECIAL_TOKENS_DICT = {'pad_token': '<pad>'}
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    print("Finished loading tokenizer")

    return tokenizer


def deduce_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device


def create_pef_config(adapter_name: str):

    if adapter_name == 'prefix_tuning':
        config = PrefixTuningConfig(flat=False, prefix_length=8)
    elif adapter_name == 'bottleneck_adapter':
        config = AdapterConfig(mh_adapter=True, output_adapter=True,
                               reduction_factor=16, non_linearity="relu")
    elif adapter_name == 'compacter':
        config = CompacterConfig()
    elif adapter_name == 'lora':
        config = LoRAConfig(r=8, alpha=16)
    elif adapter_name == 'ia3':
        config = IA3Config()

    return config


def create_T5_model(model_name: str, tokenizer: T5Tokenizer, adapter_name: str) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, output_hidden_states=True)
    model.resize_token_embeddings(len(tokenizer))

    model.add_adapter(adapter_name, config=create_pef_config(adapter_name))
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)

    model = model.to(device)

    print("Finished loading model")

    return model


def create_optimizer(model: T5ForConditionalGeneration) -> Adafactor:
    return Adafactor(
        model.parameters(),
        lr=0.0001,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )


# Create the parser
parser = argparse.ArgumentParser(description='p-tuning')

# Add the positional argument
parser.add_argument('-m', '--model_name', type=str,
                    help='short name of T5 model(large|xl|xxl)')
parser.add_argument('-e', '--exp_id', type=int, help='experiment id')

parser.add_argument('-p', '--process_id', type=int, help='id of the process')
parser.add_argument('-f', '--folder', type=str, help='folder for output')
parser.add_argument('-g', '--gpu_name', type=str,
                    help='name of the gpu that will be used')

# Parse the arguments
args = parser.parse_args()

device = deduce_device()
model_name = get_model_name(args.model_name)
adapter_name = 'prefix_tuning'  # bottleneck_adapter
tuning_name = 'softprompt'
best_em_score = 0.0

tokenizer = create_tokenizer(model_name=model_name)

print_gpu_utilization()
training_elems = TrainingElements(
    create_T5_model(model_name, tokenizer,
                    adapter_name), tokenizer, torch.cuda.amp.GradScaler(),
    lambda model: create_optimizer(model))
print_gpu_utilization()

training_config = TrainingConfig(
    model_name=model_name,
    gradient_accumulation_steps=2 if args.gpu_name == '40g' else 1,
    batch_size=16 if args.gpu_name == '40g' else 32,
    # math.ceil(50000 / (len(train_set)//32))
    gpu_stat_every=500, evaluation_every=1, num_gpus=get_number_of_gpus(),
    device=device, experiment_id=args.exp_id, epochs=100
)

training_data = TrainingData(config=training_config, tokenizer=tokenizer)

print(f'{model_name=} {adapter_name} {create_pef_config(adapter_name)}\
      {training_config.batch_size=} {training_config.epochs=}')


def main():

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MasterThesis",
        id=str(args.process_id),
        name=f'MT-T5_{args.model_name}-{tuning_name}-{args.process_id}',   
        # track hyperparameters and run metadata
        config=vars(training_config)
    )

    print("Training started...")
    for e in range(1, training_config.epochs):

        training_elems.model.train()
        torch.cuda.empty_cache()

        losses = []

        start = timer()
        for batch_idx, train_batch in enumerate(training_data.train_loader, 1):
            need_to_optimize = ((batch_idx + 1) % training_config.gradient_accumulation_steps ==
                                0) or (batch_idx + 1 == len(training_data.train_loader))
            loss = train_step(training_elements=training_elems, 
                config=training_config, train_batch=train_batch, 
                batch_idx=batch_idx, need_to_optimize=need_to_optimize)

            losses.append(loss)
        end = timer()

        print(f'loss={sum(losses)/len(losses)}')
        
        print(f'{e} took ', timedelta(seconds=end-start))
        
        wandb.log({'epoch': e, 'elapsed_time': timedelta(seconds=end-start)})

        validate(e, sum(losses)/len(losses), args.folder)

    print("\n============================\n")
    print(f'{best_em_score=}')

    wandb.log({'best_em_score': best_em_score})
    wandb.finish()


if __name__ == "__main__":
    main()
