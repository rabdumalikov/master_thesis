import wandb
import torch
import common_utils

import torch
import torch.nn as nn

from utils import *
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5_finetuning_class import Finetuning

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


class PromptTuning(Finetuning):

    def __init__(self, config: TrainingConfig, checkpoint: str = None):

        self.prompt_len = 100 # from paper: 'going beyong 20 prompt tokens only yields marginal gains'.

        def postprocessing(source):
            fake_prompt = torch.ones((source['input_ids'].size(0), self.prompt_len),
                                        dtype=source['input_ids'].dtype)

            source['input_ids'] = torch.cat(
                [fake_prompt, source['input_ids']], axis=1)[:, :512]
            source['attention_mask'] = torch.cat(
                [fake_prompt, source['attention_mask']], axis=1)[:, :512]

            return source

        super().__init__(config, checkpoint, postprocessing=postprocessing)
        

    def create_T5_model(self, checkpoint: str) -> T5ForConditionalGeneration:

        model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_name) 

        # Freeze LM
        for param in model.parameters():
            param.requires_grad = False

        prompt_emb = PromptEmbedding(model.get_input_embeddings(),
                                    prompt_length=self.prompt_len,
                                    initialize_from_vocab=True)

        if checkpoint is not None:
            print(f"Loading from checkpoint={checkpoint}")

            state_dict = torch.load(f'{checkpoint}/model.pth')            
            state_dict['wte.weight'] = model.get_input_embeddings().weight

            prompt_emb.load_state_dict( state_dict )
        
        model.set_input_embeddings(prompt_emb)

        print("Finished loading model")

        return model

    def train(self):
        self.config.closure_to_save_model = PromptTuning.custom_save_model
        return super().train()

    @staticmethod
    def get_aliases():
        return ['promptuning']

    def get_local_aliases(self):
        return PromptTuning.get_aliases()
        
    @staticmethod
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