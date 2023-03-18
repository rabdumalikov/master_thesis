import wandb
import torch
import common_utils

import torch
import torch.nn as nn

from utils import *
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration

class PromptTuning(nn.Module):
    """
    """
    def __init__(self, pretrained_config, prompt_len=20, hidden_dim=256):
        super().__init__()
        
        # Config of Pre-Trained LM
        self.pretrained_config=pretrained_config
        
        # torch.tensor([0, 1, 2, .. , prompt_len-1])
        self.pre_prompt=torch.arange(prompt_len)
        # Embedding
        self.embd=nn.Embedding(num_embeddings=prompt_len, embedding_dim=pretrained_config.d_model)
        # Reparameterization
        self.reparam=nn.Sequential(
            nn.Linear(pretrained_config.d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, pretrained_config.d_model)
        )
        
    def forward(self, batch_size, device):
        # Shape: batch_size, prompt_len
        prompt=self.pre_prompt.unsqueeze(0).expand(batch_size, -1).to(device)
        # Shape: batch_size, prompt_len, d_model
        prompt=self.embd(prompt)
        # Shape: batch_size, prompt_len, d_model
        prompt=self.reparam(prompt)
        
        return prompt

def create_T5_model(config: TrainingConfig, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(
        'google/t5-xl-lm-adapt')#, device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.cuda()

    # Freeze LM
    for param in model.parameters():
        param.requires_grad = False

    # Prompt Config
    prompt_len = 100
    prompt_model = PromptTuning(pretrained_config=model.config, prompt_len=prompt_len).to(config.device)

    print("Finished loading model")

    return model, prompt_model

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

    model, prompt_model = create_T5_model(config, tokenizer)

    training_elems = TrainingElements(
        model, tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: create_optimizer(model),
        prompt_model) 

    print_gpu_utilization()

    training_data = TrainingData(config=config, tokenizer=tokenizer, allowed_test_sets=['cf'])

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

                prompt = training_elems.prompt_model(batch_size=train_batch[0].size(
                    0), device=config.device)

                need_to_optimize = ((batch_idx + 1) % config.gradient_accumulation_steps ==
                                    0) or (batch_idx + 1 == len(training_data.train_loader))
                loss = train_step(training_elements=training_elems,
                                config=config, train_batch=train_batch,
                                batch_idx=batch_idx, need_to_optimize=need_to_optimize,
                                prompt=prompt)

                losses.append(loss)

                if len(losses) > 5:
                    break

        loss = sum(losses)/len(losses)

        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, sum(losses)/len(losses),
                                 config.model_saving_folder,
                                 best_em_score)
    return best_em_score