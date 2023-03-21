import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration


class CustomTrainingData:
    def __init__(self, config: TrainingConfig,
                 allowed_test_sets: List[int] = ['f', 'cf', 'a(e)', 'a(r)'], **kwargs):

        test_mapping = {'factual': 'f', 'counterfactual': 'cf',
                        'closed_book': 'a(e)', 'random_context': 'a(r)'}

        train_set, val_set, test_set = get_data(
            config.dataset_type, test_mapping)

        print(
            f'TrainingData: {len(train_set)=} {len(val_set)=} {len(test_set)=}')

        self.f_train_loader = DataLoader(PandasDataset(train_set[train_set['type'] == 'factual']), 
            collate_fn=lambda inp: collate_fn(inp, **kwargs), batch_size=config.batch_size, num_workers=4, pin_memory=True)


        sampler = RandomSampler(PandasDataset(
            train_set[train_set['type'] == 'counterfactual']), replacement=True)

        self.cf_train_loader = DataLoader(PandasDataset(train_set[train_set['type'] == 'counterfactual']), sampler=sampler,
            collate_fn=lambda inp: collate_fn(inp, **kwargs), batch_size=config.batch_size, num_workers=4, pin_memory=True)

        print(len(self.f_train_loader), len(self.cf_train_loader))

        self.val_loader = DataLoader(PandasDataset(val_set), collate_fn=lambda inp: collate_fn(
            inp, **kwargs), batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)

        self.test_loaders = {}
        for k in test_set:
            if k not in allowed_test_sets:
                continue

            self.test_loaders[k] = DataLoader(PandasDataset(test_set[k]), collate_fn=lambda inp: collate_fn(
                inp, **kwargs), batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)



def get_aliases():
    return ['adversarial_training']


def create_T5_model(model_name: str, tokenizer: T5Tokenizer, checkpoint: str) -> T5ForConditionalGeneration:

    model_name = checkpoint if checkpoint else model_name

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Finished loading model")

    return model


def create_stuff(config: TrainingConfig, checkpoint: str = None):
    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

    print_gpu_utilization()

    training_elems = TrainingElements(
        create_T5_model(
            config.model_name, tokenizer, checkpoint), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model))

    print_gpu_utilization()

    training_data = CustomTrainingData(config=config, tokenizer=tokenizer)

    return training_elems, training_data


def train_step(training_elements: TrainingElements, config: TrainingConfig,
               train_batch, counterfactual_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

    torch.cuda.empty_cache()

    cf_src_ids = counterfactual_batch[0].to(0)
    cf_src_am = counterfactual_batch[1].to(0)
    cf_trg_ids = counterfactual_batch[2].to(0)

    cf_lm_labels = cf_trg_ids.clone().detach()
    cf_lm_labels[cf_trg_ids ==
                   training_elements.tokenizer.pad_token_id] = -100

    # target = training_elements.tokenizer.batch_decode(
    #     cf_trg_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # print(f'TestCF: {target}')

    src_ids = train_batch[0].to(0)
    src_am = train_batch[1].to(0)
    trg_ids = train_batch[2].to(0)

    lm_labels = trg_ids.clone().detach()
    lm_labels[trg_ids == training_elements.tokenizer.pad_token_id] = -100

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

        loss = loss1 + 0.25 * loss2 # 0.25 comes from undersensitivity paper

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


def run(config: TrainingConfig, alias: str):

    training_elems, training_data = create_stuff(config)

    print("Training started...")
    print(f'{config.model_name=} {config.batch_size=} {config.epochs=}')

    best_em_score = 0.0
    for e in range(1, config.epochs):

        training_elems.model.train()
        torch.cuda.empty_cache()

        losses = []

        steps = get_number_training_steps(e, len(training_data.f_train_loader), config.batch_size )
        with TimeMeasure(epoch=e, steps=steps):
            for batch_idx, train_batch in enumerate(training_data.f_train_loader, 1):
                need_to_optimize = ((batch_idx + 1) % config.gradient_accumulation_steps ==
                                    0) or (batch_idx + 1 == len(training_data.f_train_loader))

                for batch_idx, batch in enumerate(training_data.cf_train_loader, 1):
                    counterfactual_batch = batch

                    break

                loss = train_step(training_elements=training_elems,
                                  config=config, train_batch=train_batch, counterfactual_batch=counterfactual_batch,
                                  batch_idx=batch_idx, need_to_optimize=need_to_optimize)

                losses.append(loss)

        loss = sum(losses)/len(losses)

        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, loss, config.model_saving_folder, best_em_score)

    return best_em_score
