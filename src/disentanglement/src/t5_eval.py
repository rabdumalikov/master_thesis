
import pandas as pd
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


def create_stuff(config: TrainingConfig):
    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

    print_gpu_utilization()

    training_elems = TrainingElements(
        create_T5_model(
            config.model_name, tokenizer), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model))

    print_gpu_utilization()

    training_data = TrainingData(
        config=config, tokenizer=tokenizer)

    return training_elems, training_data


def find_best_checkpoint(id: int):
    dir_path = f'.builds/{id}/models'

    checkpoint_name = ''
    checkpoint_id = -1
    # iterate over all the directories in the specified directory
    for dir_name in os.listdir(dir_path):

        if os.path.isdir(os.path.join(dir_path, dir_name)):
            id = int(dir_name.split('_')[-1])
            if id > checkpoint_id:
                checkpoint_id = id
                checkpoint_name = dir_name

            print(dir_name)

    return dir_path + '/' + checkpoint_name


def main():

    model_name = find_best_checkpoint(197)

    config = TrainingConfig(model_name=model_name,
                            dataset_type='s(f)', batch_size=32, eval_batch_size=32)

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

    torch.cuda.empty_cache()

    return evaluate(training_elems, config, training_data.test_loader, verbose=True)


if __name__ == '__main__':
    main()
