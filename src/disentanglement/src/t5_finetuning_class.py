import math
import wandb
import torch
import common_utils

from utils import *
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

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

        #model = model.to(self.config.device)

        # TODO: optimizer changed. utils.py
        self.training_elems = TrainingElements(model, tokenizer, self.create_optimizer(model))

        print_gpu_utilization()


    def create_optimizer(self, model: T5ForConditionalGeneration):
        return common_utils.create_optimizer(model, self.config.tuning_settings['learning_rate'],
            self.config.tuning_settings['type_of_optimizer'])

    def create_T5_model(self, checkpoint: str) -> T5ForConditionalGeneration:

        model_name = checkpoint if checkpoint else self.config.model_name
        
        config = T5Config.from_pretrained(model_name)
        with init_empty_weights():
            model = T5ForConditionalGeneration(config=config)

        device_map = infer_auto_device_map(model, max_memory={0: "25GiB", 1: "25GiB", 2: "25GiB", 3: "25GiB"},  no_split_module_classes=["T5Block"], dtype="float16")


        model = T5ForConditionalGeneration.from_pretrained(
            model_name, 
            device_map=device_map,
            offload_folder="offload",
            offload_state_dict=True,
            torch_dtype=torch.float16)
            
        #model = T5ForConditionalGeneration.from_pretrained(
        #    model_name, device_map='auto')

        print(model.hf_device_map)
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

        best_val_loss = 0.0 if self.config.val_accuracy or self.config.tuning_method == 'finetuning' else math.inf
        
        for e in range(1, self.config.epochs):

            self.training_elems.model.train()
            torch.cuda.empty_cache()

            losses = []

            steps = get_number_training_steps(e, len(self.training_data.train_loader), self.config.batch_size )

            if self.config.skip_train == False:
                with TimeMeasure(epoch=e, steps=steps):
                    for batch_idx, train_batch in enumerate(self.training_data.train_loader, 1):
                        loss = train_step(training_elements=self.training_elems,
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
