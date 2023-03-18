import wandb
import torch
import common_utils

import torch
import torch.nn as nn

from utils import *
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration

class PROMPTEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(wte.weight.size(1), n_tokens).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        # More robust strategy would be to check that all first tokens are ones
        # Thus it is input.

        if tokens.size(1) < self.n_tokens:
            return self.wte(tokens)
        else:
            input_embedding = self.wte(tokens[:, self.n_tokens:])
            learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
            return torch.cat([learned_embedding, input_embedding], 1)

def create_T5_model(config: TrainingConfig, tokenizer: T5Tokenizer, prompt_len: int = 100) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(
        'google/t5-xl-lm-adapt')#, device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    #Freeze LM
    for param in model.parameters():
       param.requires_grad = False


    prompt_emb = PROMPTEmbedding(model.get_input_embeddings(), 
                        n_tokens=prompt_len, 
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
    tokenizer = common_utils.create_tokenizer(model_name='google/t5-xl-lm-adapt')

    print_gpu_utilization()

    prompt_len = 100
    model = create_T5_model(config, tokenizer, prompt_len)

    training_elems = TrainingElements(
        model, tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: create_optimizer(model)) 

    print_gpu_utilization()

    def postprocessing( source ):
        fake_prompt = torch.ones((source['input_ids'].size(0), prompt_len),
                   dtype=source['input_ids'].dtype)

        source['input_ids'] = torch.cat([fake_prompt, source['input_ids']], axis=1)[:,:512]
        source['attention_mask'] = torch.cat([fake_prompt, source['attention_mask']], axis=1)[:,:512]
        

        return source

    training_data = TrainingData(config=config, allowed_test_sets=['cf'], 
        tokenizer=tokenizer, closure=None, postprocessing=postprocessing)

    return training_elems, training_data


def run(config: TrainingConfig):

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

    best_em_score = 0.0
    for e in range(1, config.epochs):

        training_elems.model.train()
        torch.cuda.empty_cache()

        losses = []

        with TimeMeasure(epoch=e) as tm:
            for batch_idx, train_batch in enumerate(training_data.train_loader, 1):

                need_to_optimize = ((batch_idx + 1) % config.gradient_accumulation_steps ==
                                    0) or (batch_idx + 1 == len(training_data.train_loader))
                loss = train_step(training_elements=training_elems,
                                config=config, train_batch=train_batch,
                                batch_idx=batch_idx, need_to_optimize=need_to_optimize
                            )

                losses.append(loss)

                # if len(losses) >= 10:
                #     break

        loss = sum(losses)/len(losses)

        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, sum(losses)/len(losses),
                                 config.model_saving_folder,
                                 best_em_score)
    return best_em_score