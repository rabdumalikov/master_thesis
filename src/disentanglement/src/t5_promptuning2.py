import wandb
import torch
import common_utils

import torch
import torch.nn as nn

from utils import *
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration


class PromptEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 prompt_length: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(PromptEmbedding, self).__init__()
        self.wte = wte
        self.prompt_length = prompt_length
        self.learned_embedding = nn.parameter.Parameter(
            self.initialize_embedding(wte,
                                      prompt_length,
                                      random_range,
                                      initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             prompt_length: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:prompt_length].clone().detach()

        return torch.FloatTensor(wte.weight.size(1),
                                 prompt_length).uniform_(-random_range, random_range)

    def forward(self, tokens):
        # We need to identify whether fake_prompt was added or not, because if it
        # is not then we should just apply T5's embedding layer(since we are replacing
        # T5 original embedding layer with ours). fake_prompt is
        # a dummy payload of all 1's. It is used just to book the space.

        if (tokens[:, : self.prompt_length] == 1).all().item():
            input_embedding = self.wte(tokens[:, self.prompt_length:])
            learned_embedding = self.learned_embedding.repeat(
                input_embedding.size(0), 1, 1)
            return torch.cat([learned_embedding, input_embedding], 1)
        else:
            return self.wte(tokens)

def custom_save_model(training_elements: TrainingElements, em_score: float, loss: float, e: int, model_name: str, folder: str):
    ext = model_name.split('-')[-1]

    print(em_score, ext, loss, e, model_name)

    checkpoint_path = f'{folder}/checkpoint_{e}'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    with open(f'{checkpoint_path}/results.txt', 'w') as f:
        f.write(f'epoch={e}\nloss={loss}\nem={round(em_score,3)}')

    prompt_embed = training_elements.model.get_input_embeddings()

    torch.save(prompt_embed.state_dict(), f'{checkpoint_path}/model.pth')

def get_aliases():
    return ['promptuning']


def create_T5_model(config: TrainingConfig, tokenizer: T5Tokenizer, prompt_len: int = 100) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(
        config.model_name)  # , device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Freeze LM
    for param in model.parameters():
        param.requires_grad = False

    prompt_emb = PromptEmbedding(model.get_input_embeddings(),
                                 prompt_length=prompt_len,
                                 initialize_from_vocab=True)
    model.set_input_embeddings(prompt_emb)
    model.cuda()

    print("Finished loading model")

    return model


def create_optimizer(model: T5ForConditionalGeneration) -> Adafactor:
    return Adafactor(
        model.parameters(),
        lr=3e-1,
        beta1=0.8,
        weight_decay=1e-5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )


def create_stuff(config: TrainingConfig):
    tokenizer = common_utils.create_tokenizer(
        model_name=config.model_name)

    print_gpu_utilization()

    prompt_len = 100
    model = create_T5_model(config, tokenizer, prompt_len)

    training_elems = TrainingElements(
        model, tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: create_optimizer(model))

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


def run(config: TrainingConfig, alias: str):

    config.closure_to_save_model = custom_save_model

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

    best_em_score = 0.0
    for e in range(1, config.epochs):

        training_elems.model.train()
        torch.cuda.empty_cache()

        losses = []

        with TimeMeasure(epoch=e):
            for batch_idx, train_batch in enumerate(training_data.train_loader, 1):

                need_to_optimize = ((batch_idx + 1) % config.gradient_accumulation_steps ==
                                    0) or (batch_idx + 1 == len(training_data.train_loader))
                loss = train_step(training_elements=training_elems,
                                  config=config, train_batch=train_batch,
                                  batch_idx=batch_idx, need_to_optimize=need_to_optimize
                                  )

                losses.append(loss)


        loss = sum(losses)/len(losses)

        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, sum(losses)/len(losses),
                                 config.model_saving_folder,
                                 best_em_score)
    return best_em_score
