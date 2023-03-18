import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration


def create_T5_model(model_name: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Finished loading model")

    return model


def preprocess_input(input: pd.DataFrame):
    return ("Using context: " + "'" + input['context'] + "'" + ". Answer following question: " + "'" + input['question'] + "'" + " </s>").tolist()


def create_stuff(config: TrainingConfig):
    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

    print_gpu_utilization()

    training_elems = TrainingElements(
        create_T5_model(
            config.model_name, tokenizer), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model))

    print_gpu_utilization()

    training_data = TrainingData(
        config=config, tokenizer=tokenizer, closure=preprocess_input)

    return training_elems, training_data


def run(config: TrainingConfig):

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

    torch.cuda.empty_cache()
    
    return evaluate(training_elems, config, training_data.test_loader, verbose=True)
