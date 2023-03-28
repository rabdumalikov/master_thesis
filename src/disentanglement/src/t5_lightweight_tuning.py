import wandb
import torch
import common_utils

from utils import *
from transformers import AdamW, get_scheduler
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, LoRAConfig


def get_aliases():
    return ['prefix_tuning', 'bottleneck_adapter', 'lora']

def create_pef_config(adapter_name: str):

    if adapter_name == 'prefix_tuning':
        return PrefixTuningConfig(flat=True, prefix_length=100) # based on prefix-tuning paper
    elif adapter_name == 'bottleneck_adapter':
        return AdapterConfig(mh_adapter=True, output_adapter=True,
                             reduction_factor=16, non_linearity="relu")
    elif adapter_name == 'lora':
        return LoRAConfig(r=16, alpha=32) # based on adapters paper
    else:
        return None

def create_T5_model(model_name: str, tokenizer: T5Tokenizer, adapter_name: str, device: torch.device, checkpoint: str) -> T5ForConditionalGeneration:
    
    model_name = checkpoint if checkpoint else model_name

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, output_hidden_states=True)
    model.resize_token_embeddings(len(tokenizer))

    if checkpoint is None:
        model.add_adapter(adapter_name, config=create_pef_config(adapter_name))
        model.train_adapter(adapter_name)
    
    model.set_active_adapters(adapter_name)

    model = model.to(device)

    print("Finished loading model")

    return model

def create_stuff(config: TrainingConfig, checkpoint: str = None):

    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)
    training_data = TrainingData(config=config, tokenizer=tokenizer)

    print_gpu_utilization()
    training_elems = TrainingElements(
        create_T5_model(config.model_name, tokenizer,
                        config.tuning_method, config.device, checkpoint), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model) )
    print_gpu_utilization()

    return training_elems, training_data

def run(config: TrainingConfig, adapter_name: str) -> float:

    if adapter_name not in get_aliases():
        print("Wrong alias were used! Thus terminate.")
        return 0.0

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {adapter_name} {create_pef_config(adapter_name)}\
      {config.batch_size=} {config.epochs=}')

    best_em_score = 0.0

    for e in range(1, config.epochs):

        training_elems.model.train()
        torch.cuda.empty_cache()

        losses = []

        steps = get_number_training_steps(e, len(training_data.train_loader), config.batch_size )
        with TimeMeasure(epoch=e, steps=steps):
            for batch_idx, train_batch in enumerate(training_data.train_loader, 1):
                need_to_optimize = ((batch_idx + 1) % config.gradient_accumulation_steps ==
                                    0) or (batch_idx + 1 == len(training_data.train_loader))

                loss = train_step(training_elements=training_elems,
                                  config=config, train_batch=train_batch,
                                  batch_idx=batch_idx, need_to_optimize=need_to_optimize)

                losses.append(loss)

        loss = sum(losses)/len(losses)


        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, sum(losses)/len(losses), config.model_saving_folder, best_em_score)

    return best_em_score
