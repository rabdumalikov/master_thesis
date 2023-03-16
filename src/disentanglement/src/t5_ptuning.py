import os
import wandb
import torch
import argparse

from utils import *
from common_utils import *
from datetime import timedelta
from timeit import default_timer as timer
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, CompacterConfig, LoRAConfig, IA3Config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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


def run( training_config: TrainingConfig ):

    print("Training started...")
    print(f'{model_name=} {adapter_name} {create_pef_config(adapter_name)}\
      {training_config.batch_size=} {training_config.epochs=}')

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

        validate(training_elems, training_data, training_config, e, sum(losses)/len(losses), args.folder)

    print("\n============================\n")
    print(f'{best_em_score=}')

    wandb.log({'best_em_score': best_em_score})
