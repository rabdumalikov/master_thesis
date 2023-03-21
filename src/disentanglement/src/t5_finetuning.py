import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration


def get_aliases():
    return ['finetuning']


def create_T5_model(model_name: str, tokenizer: T5Tokenizer, checkpoint: str) -> T5ForConditionalGeneration:

    model_name = checkpoint if checkpoint else model_name
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Finished loading model")

    return model


def create_stuff(config: TrainingConfig, checkpoint: str = None):
    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

    print_gpu_utilization()

    training_elems = TrainingElements(
        create_T5_model(
            config.model_name, tokenizer, checkpoint), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model))

    print_gpu_utilization()

    training_data = TrainingData(config=config, tokenizer=tokenizer)

    return training_elems, training_data

def run(config: TrainingConfig, alias: str):

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

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
                                 e, loss, config.model_saving_folder, best_em_score)

    return best_em_score
