import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration


def create_T5_model(model_name: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map='balanced')
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    print("Finished loading model")

    return model


def create_stuff(config: TrainingConfig):
    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

    print_gpu_utilization()

    training_elems = TrainingElements(
        create_T5_model(
            config.model_name, tokenizer), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model))

    print_gpu_utilization()

    training_data = TrainingData(config=config, tokenizer=tokenizer)

    return training_elems, training_data


def train_step(training_elements: TrainingElements, config: TrainingConfig,
               train_batch, test_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

    torch.cuda.empty_cache()

    test_src_ids = test_batch[0].to(0)
    test_src_am = test_batch[1].to(0)
    test_trg_ids = test_batch[2].to(0)
    test_lm_labels = test_trg_ids.clone().detach()
    test_lm_labels[test_trg_ids == training_elements.tokenizer.pad_token_id] = -100

    # target = training_elements.tokenizer.batch_decode(
    #     test_trg_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
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
            input_ids=test_src_ids,
            attention_mask=test_src_am,
            labels=test_lm_labels.to(f'cuda:{config.num_gpus-1}'),
            **kwargs
        )[0]

        loss = loss1 + 0.5 * loss2

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

                for batch_idx, batch in enumerate(training_data.test_loaders['cf'], 1):
                    test_batch = batch

                    break

                loss = train_step(training_elements=training_elems,
                            config=config, train_batch=train_batch, test_batch=test_batch,
                            batch_idx=batch_idx, need_to_optimize=need_to_optimize)
                    
                losses.append(loss)

                # if len(losses) > 10:
                #     break

        loss = sum(losses)/len(losses)

        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, loss, config.model_saving_folder, best_em_score)

    return best_em_score