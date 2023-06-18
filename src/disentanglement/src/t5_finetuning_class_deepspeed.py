import math
import wandb
import torch
import deepspeed
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Step 3: Model parallelism configuration
ds_config = {
    "model": {
        "activation_checkpointing": True,  # Enable activation checkpointing for memory optimization
        "tensor_model_parallel_size": 4,  # Split the model across 2 devices
        "num_layers": 24,  # Total number of layers in your T5 model
        "num_attention_heads": 16,  # Number of attention heads in your T5 model
        "hidden_size": 1024,  # Hidden size of your T5 model
        # Add any other T5-specific configuration options
    },
    "optimizer": {
        "type": "Adafactor",  # Use AdamW optimizer
        "params": {
            "lr": 1e-5,  # Learning rate
        },
    },
}

class FinetuningDeepSpeed:

    @staticmethod
    def get_aliases():
        return ['finetuning_deepspeed']

    def get_local_aliases(self):
        return FinetuningDeepSpeed.get_aliases()

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

        model, optimizer, _, _ = deepspeed.initialize(
                                    model=model,
                                    model_parameters=model.parameters(),
                                    config=ds_config)

        # TODO: optimizer changed. utils.py
        self.training_elems = TrainingElements(model, tokenizer, optimizer)

        print_gpu_utilization()

    def create_optimizer(self, model: T5ForConditionalGeneration):
        return common_utils.create_optimizer(model, self.config.tuning_settings['learning_rate'],
            self.config.tuning_settings['type_of_optimizer'])

    def create_T5_model(self, checkpoint: str) -> T5ForConditionalGeneration:

        model_name = checkpoint if checkpoint else self.config.model_name
        
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map='balanced')

        num_param = count_parameters(model)
        print(f'Fine-Tuning number of parameters is {num_param}')

        print("Finished loading model")

        return model

    def eval(self):
        torch.cuda.empty_cache()

        for k in self.training_data.test_loaders:
            acc = evaluate(self.training_elems, self.config, self.training_data.test_loaders[k], verbose=True)
            print(f'{k=} {acc=}')

    def train(self):
        
        print(f"Training started...")
        print(f'{self.config.model_name=} {self.config.tuning_method=} {self.config.batch_size=} {self.config.epochs=}')

        best_val_loss = 0.0 if self.config.val_accuracy or self.config.tuning_method == 'finetuning_deepspeed' else math.inf
        
        for e in range(1, self.config.epochs):

            self.training_elems.model.train()
            torch.cuda.empty_cache()

            losses = []

            steps = get_number_training_steps(e, len(self.training_data.train_loader), self.config.batch_size )

            if self.config.skip_train == False:
                with TimeMeasure(epoch=e, steps=steps):
                    for batch_idx, train_batch in enumerate(self.training_data.train_loader, 1):
                        loss = deepspeed_train_step(training_elements=self.training_elems,
                                        config=self.config, train_batch=train_batch,
                                        batch_idx=batch_idx, need_to_optimize=self.need_to_optimize(batch_idx))

                        losses.append(loss)
                loss = sum(losses)/len(losses)
            else:
                loss = 0.5
            
            print(f'{loss=}')

            best_val_loss = validate(self.training_elems, self.training_data, self.config,
                                    e, loss, self.config.model_saving_folder, best_val_loss, verbose=False)

        return best_val_loss

    def need_to_optimize(self, batch_idx):
        return ((batch_idx + 1) % self.config.gradient_accumulation_steps ==
                    0) or (batch_idx + 1 == len(self.training_data.train_loader))
