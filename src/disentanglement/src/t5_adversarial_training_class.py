import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5_finetuning_class import Finetuning


class CustomTrainingData(TrainingData):
    def __init__(self, config: TrainingConfig,
                 allowed_test_sets: List[int] = ['f', 'cf', 'a(e)', 'a(r)'], **kwargs):

        super().__init__(config, allowed_test_sets, **kwargs)

        train_set = self.train_loader.dataset.get_dataframe()

        self.f_train_loader = DataLoader(PandasDataset(train_set[train_set['type'] == 'factual']), 
            collate_fn=lambda inp: collate_fn(inp, max_source_input_len=256, **kwargs), batch_size=config.batch_size, num_workers=4, pin_memory=True)

        sampler = RandomSampler(PandasDataset(
            train_set[train_set['type'] == 'counterfactual']), replacement=True)

        self.cf_train_loader = DataLoader(PandasDataset(train_set[train_set['type'] == 'counterfactual']), sampler=sampler,
            collate_fn=lambda inp: collate_fn(inp, max_source_input_len=256, **kwargs), batch_size=config.batch_size, num_workers=4, pin_memory=True)

        print(len(self.f_train_loader), len(self.cf_train_loader))


class AdversarialTraining(Finetuning):
    @staticmethod
    def get_aliases():
        return ['adversarial_training']

    def get_local_aliases(self):
        return AdversarialTraining.get_aliases()

    def __init__(self, config: TrainingConfig, checkpoint: str = None):
        super().__init__(config, checkpoint)

        self.training_data = CustomTrainingData(config=self.config, tokenizer=self.training_elems.tokenizer)

    def train(self):
        
        print("Training started...")
        print(f'{self.config.model_name=} {self.config.batch_size=} {self.config.epochs=}')

        best_em_score = 0.0
        for e in range(1, self.config.epochs):

            self.training_elems.model.train()
            torch.cuda.empty_cache()

            losses = []

            steps = get_number_training_steps(e, len(self.training_data.f_train_loader), self.config.batch_size )
            with TimeMeasure(epoch=e, steps=steps):
                for batch_idx, train_batch in enumerate(self.training_data.f_train_loader, 1):

                    for batch_idx, batch in enumerate(self.training_data.cf_train_loader, 1):
                        counterfactual_batch = batch
                        break

                    loss = AdversarialTraining.train_step(training_elements=self.training_elems,
                                    config=self.config, train_batch=train_batch, counterfactual_batch=counterfactual_batch,
                                    batch_idx=batch_idx, need_to_optimize=self.need_to_optimize(batch_idx))

                    losses.append(loss)

            loss = sum(losses)/len(losses)

            print(f'{loss=}')

            best_em_score = validate(self.training_elems, self.training_data, self.config,
                                    e, loss, self.config.model_saving_folder, best_em_score)

        return best_em_score

    @staticmethod
    def train_step(training_elements: TrainingElements, config: TrainingConfig,
                train_batch, counterfactual_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

        torch.cuda.empty_cache()

        cf_src_ids, cf_src_am, cf_lm_labels = unroll_batch( counterfactual_batch, 
            config.device, training_elements.tokenizer.pad_token_id )

        src_ids, src_am, lm_labels = unroll_batch( train_batch, 
            config.device, training_elements.tokenizer.pad_token_id )

        with autocast(dtype=torch.bfloat16, enabled=config.FP16):
            loss1 = training_elements.model(
                input_ids=src_ids,
                attention_mask=src_am,
                labels=lm_labels.to(f'cuda:{config.num_gpus-1}'),
                **kwargs
            )[0]

            loss2 = training_elements.model(
                input_ids=cf_src_ids,
                attention_mask=cf_src_am,
                labels=cf_lm_labels.to(f'cuda:{config.num_gpus-1}'),
                **kwargs
            )[0]

            reguralizer = 0.25
            
            loss = loss1 + reguralizer * loss2 # 0.25 comes from undersensitivity paper

        # normalize loss to account for batch accumulation
        loss = loss / config.gradient_accumulation_steps

        training_elements.scaler.scale(loss).backward()

        if need_to_optimize:
            training_elements.scaler.step(training_elements.optimizer)
            training_elements.scaler.update()
            training_elements.optimizer.zero_grad()

        if batch_idx % config.gpu_stat_every == 0:
            print_gpu_utilization()
            torch.cuda.empty_cache()

        return loss.item()

