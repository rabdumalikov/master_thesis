import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Finetuning:

    @staticmethod
    def get_aliases():
        return ['finetuning']

    def get_local_aliases(self):
        return Finetuning.get_aliases()

    def __init__(self, config: TrainingConfig, checkpoint: str = None, closure = None, postprocessing = None):

        if config.tuning_method not in self.get_local_aliases():
            raise Exception(f"Given alias {config.tuning_method} isn't supported. Supported are {self.get_local_aliases()}")

        self.config = config

        tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

        self.training_data = TrainingData(config=config, tokenizer=tokenizer,
            closure=closure, postprocessing=postprocessing)

        print_gpu_utilization()

        # TODO: each class would specify their own method to create model
        model = self.create_T5_model(checkpoint)

        # NOTE: PromptTuning will break if I uncomment this
        #model.resize_token_embeddings(len(tokenizer))

        if self.config.gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model = model.to(self.config.device)

        # TODO: optimizer changed. utils.py
        self.training_elems = TrainingElements(model, tokenizer, self.create_optimizer(model))

        print_gpu_utilization()


    def create_optimizer(self, model: T5ForConditionalGeneration):
        return common_utils.create_optimizer(model)

    def create_T5_model(self, checkpoint: str) -> T5ForConditionalGeneration:

        model_name = checkpoint if checkpoint else self.config.model_name
        
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map='balanced')

        print("Finished loading model")

        return model

    def eval(self):
        torch.cuda.empty_cache()

        for k in self.training_data.test_loaders:
            acc = evaluate(self.training_elems, self.config, self.training_data.test_loaders[k], verbose=True)
            print(f'{k=} {acc=}')

    def train(self):

        print("Training started...")
        print(f'{self.config.model_name=} {self.config.tuning_method=} {self.config.batch_size=} {self.config.epochs=}')

        best_em_score = 0.0
        for e in range(1, self.config.epochs):

            self.training_elems.model.train()
            torch.cuda.empty_cache()

            losses = []

            steps = get_number_training_steps(e, len(self.training_data.train_loader), self.config.batch_size )
            
            with TimeMeasure(epoch=e, steps=steps):
                for batch_idx, train_batch in enumerate(self.training_data.train_loader, 1):

                    loss = train_step(training_elements=self.training_elems,
                                    config=self.config, train_batch=train_batch,
                                    batch_idx=batch_idx, need_to_optimize=self.need_to_optimize(batch_idx))

                    losses.append(loss)

            loss = sum(losses)/len(losses)

            print(f'{loss=}')

            best_em_score = validate(self.training_elems, self.training_data, self.config,
                                    e, loss, self.config.model_saving_folder, best_em_score, verbose=False)

        return best_em_score

    def need_to_optimize(self, batch_idx):
        return ((batch_idx + 1) % self.config.gradient_accumulation_steps ==
                    0) or (batch_idx + 1 == len(self.training_data.train_loader))
