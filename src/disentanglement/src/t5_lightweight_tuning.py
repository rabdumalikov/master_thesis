import wandb
import torch
import common_utils

from utils import *
from transformers import AdamW, get_scheduler
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, LoRAConfig


def get_aliases():
    return ['prefix_tuning', 'bottleneck_adapter', 'lora']


def create_pef_config(adapter_name: str):

    if adapter_name == 'prefix_tuning':
        return PrefixTuningConfig(flat=False, prefix_length=10) # based on prefix-tuning paper
    elif adapter_name == 'bottleneck_adapter':
        return AdapterConfig(mh_adapter=True, output_adapter=True,
                             reduction_factor=16, non_linearity="relu")
    elif adapter_name == 'lora':
        return LoRAConfig(r=4, alpha=32) # based on adapters paper
    else:
        return None

def get_optimizer(adapter_name: str):

    return common_utils.create_optimizer

    # if adapter_name == 'prefix_tuning':
    #     return create_optimizer_for_prefix
    # elif adapter_name == 'bottleneck_adapter':
    #     return create_optimizer_for_adapter
    # elif adapter_name == 'lora':
    #     return create_optimizer_for_lora
    # else:
    #     return None

def create_T5_model(model_name: str, tokenizer: T5Tokenizer, adapter_name: str, device: torch.device, checkpoint: str) -> T5ForConditionalGeneration:
    
    model_name = checkpoint if checkpoint else model_name

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, output_hidden_states=True)
    model.resize_token_embeddings(len(tokenizer))

    if checkpoint is None:
        model.add_adapter(adapter_name, config=create_pef_config(adapter_name))
        model.train_adapter(adapter_name)
    
    model.set_active_adapters(adapter_name)

    model = model.to(device)

    print("Finished loading model")

    return model


def create_stuff(config: TrainingConfig, checkpoint: str = None):

    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)
    training_data = CustomTrainingData(config=config, tokenizer=tokenizer)

    print_gpu_utilization()
    training_elems = TrainingElements(
        create_T5_model(config.model_name, tokenizer,
                        config.tuning_method, config.device, checkpoint), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: get_optimizer(config.tuning_method)(model) )
    print_gpu_utilization()

    return training_elems, training_data

class CustomTrainingData:
    def __init__(self, config: TrainingConfig,
                 allowed_test_sets: List[int] = ['f', 'cf', 'a(e)', 'a(r)'], **kwargs):

        test_mapping = {'factual': 'f', 'counterfactual': 'cf',
                        'closed_book': 'a(e)', 'random_context': 'a(r)'}

        train_set, val_set, test_set = get_data(
            config.dataset_type, test_mapping)

        print(
            f'TrainingData: {len(train_set)=} {len(val_set)=} {len(test_set)=}')

        self.train_loader = DataLoader(PandasDataset(train_set[train_set['type'] == 'factual']), 
            collate_fn=lambda inp: collate_fn(inp, max_source_input_len=256, **kwargs), batch_size=config.batch_size, num_workers=4, pin_memory=True)


        # sampler = RandomSampler(PandasDataset(
        #     train_set[train_set['type'] == 'counterfactual']), replacement=True)

        # self.cf_train_loader = DataLoader(PandasDataset(train_set[train_set['type'] == 'counterfactual']), sampler=sampler,
        #     collate_fn=lambda inp: collate_fn(inp, max_source_input_len=256, **kwargs), max_source_input_len=256, batch_size=config.batch_size, num_workers=4, pin_memory=True)

        #print(len(self.f_train_loader), len(self.cf_train_loader))

        self.val_loader = DataLoader(PandasDataset(val_set), collate_fn=lambda inp: collate_fn(
            inp, max_source_input_len=256, **kwargs), batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)

        self.test_loaders = {}
        for k in test_set:
            if k not in allowed_test_sets:
                continue

            self.test_loaders[k] = DataLoader(PandasDataset(test_set[k]), collate_fn=lambda inp: collate_fn(
                inp, max_source_input_len=396, **kwargs), batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)

    def to_readable_name(self, abbreviation: str):
        if abbreviation == 'f': 
            return "Factual"
        elif abbreviation == 'cf': 
            return "Counterfactual"
        elif abbreviation == 'a(e)': 
            return "Empty"
        elif abbreviation == 'a(r)': 
            return "Random"

def train_step(training_elements: TrainingElements, config: TrainingConfig,
               train_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

    torch.cuda.empty_cache()

    # cf_src_ids, cf_src_am, cf_lm_labels = unroll_batch( counterfactual_batch, 
    #     config.device, training_elements.tokenizer.pad_token_id )

    # target = training_elements.tokenizer.batch_decode(
    #     cf_trg_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # print(f'TestCF: {target}')

    src_ids, src_am, lm_labels = unroll_batch( train_batch, 
        config.device, training_elements.tokenizer.pad_token_id )

    with autocast(dtype=torch.bfloat16, enabled=config.FP16):
        loss = training_elements.model(
            input_ids=src_ids,
            attention_mask=src_am,
            labels=lm_labels.to(f'cuda:{config.num_gpus-1}'),
            **kwargs
        )[0]

        # loss2 = training_elements.model(
        #     input_ids=cf_src_ids,
        #     attention_mask=cf_src_am,
        #     labels=cf_lm_labels.to(f'cuda:{config.num_gpus-1}'),
        #     **kwargs
        # )[0]

        # reguralizer = 0.25
        
        #loss = loss1 + reguralizer * loss2 # 0.25 comes from undersensitivity paper

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



def run(config: TrainingConfig, adapter_name: str) -> float:

    if adapter_name not in get_aliases():
        print("Wrong alias were used! Thus terminate.")
        return 0.0

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {adapter_name} {create_pef_config(adapter_name)}\
      {config.batch_size=} {config.epochs=}')

    best_em_score = 0.0

    for e in range(1, config.epochs):

        training_elems.model.train()
        torch.cuda.empty_cache()

        losses = []

        steps = get_number_training_steps(e, len(training_data.train_loader), config.batch_size )
        with TimeMeasure(epoch=e, steps=steps):
            for batch_idx, train_batch in enumerate(training_data.train_loader, 1):
                need_to_optimize = ((batch_idx + 1) % config.gradient_accumulation_steps ==
                                    0) or (batch_idx + 1 == len(training_data.train_loader))

                # for batch_idx, batch in enumerate(training_data.cf_train_loader, 1):
                #     counterfactual_batch = batch

                #     break
                loss = train_step(training_elements=training_elems,
                                  config=config, train_batch=train_batch, # counterfactual_batch=counterfactual_batch,
                                  batch_idx=batch_idx, need_to_optimize=need_to_optimize)

                losses.append(loss)

        loss = sum(losses)/len(losses)


        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, sum(losses)/len(losses), config.model_saving_folder, best_em_score)

    return best_em_score
