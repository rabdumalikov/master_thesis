import torch
import pandas as pd
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, LoRAConfig, AutoAdapterModel


def create_T5_model(model_name: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:
    
    model = T5ForConditionalGeneration.from_pretrained(
            model_name
        )
    #model.set_active_adapters('prefix_tuning')
    #model.cuda()

    # Freeze LM
    for param in model.parameters():
        param.requires_grad = False

    prompt_emb = torch.load( find_best_checkpoint(262) + '/model.pth' )
    
    # prompt_emb = PromptEmbedding(model.get_input_embeddings(),
    #                              prompt_length=prompt_len,
    #                              initialize_from_vocab=True)
    model.set_input_embeddings(prompt_emb)
    model.cuda()

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

    def postprocessing(source):
        fake_prompt = torch.ones((source['input_ids'].size(0), prompt_len),
                                 dtype=source['input_ids'].dtype)

        source['input_ids'] = torch.cat(
            [fake_prompt, source['input_ids']], axis=1)[:, :512]
        source['attention_mask'] = torch.cat(
            [fake_prompt, source['attention_mask']], axis=1)[:, :512]

        return source

    training_data = TrainingData(
        config=config, tokenizer=tokenizer, closure=None, postprocessing=postprocessing)

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

    model_name = find_best_checkpoint(262)

    with open(model_name+'/results.txt', 'r') as f:
        print(f.readlines())

    config = TrainingConfig(model_name=model_name,
                            dataset_type='s(f)', batch_size=32,
                            num_gpus=1, gradient_accumulation_steps=1,
                            gpu_stat_every=500, 
                            evaluation_every=1, 
                            device=deduce_device(), 
                            epochs=100, 
                            model_saving_folder='',
                )
    config.model_name = 'google/t5-xl-lm-adapt'

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

    torch.cuda.empty_cache()

    for k in training_data.test_loaders:
        acc = evaluate(training_elems, config, training_data.test_loaders[k], verbose=True)
        print(f'{k=} {acc=}')


if __name__ == '__main__':
    main()
