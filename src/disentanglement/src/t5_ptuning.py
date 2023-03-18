import wandb
import torch
import common_utils

from utils import *
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, CompacterConfig, LoRAConfig, IA3Config


def create_pef_config(adapter_name: str):

    if adapter_name == 'prefix_tuning':
        config = PrefixTuningConfig(flat=False, prefix_length=1)
    elif adapter_name == 'bottleneck_adapter':
        config = AdapterConfig(mh_adapter=True, output_adapter=True,
                               reduction_factor=16, non_linearity="relu")
    elif adapter_name == 'compacter':
        config = CompacterConfig()
    elif adapter_name == 'lora':
        config = LoRAConfig(r=8, alpha=16)
    elif adapter_name == 'ia3':
        config = IA3Config()

    return config


def create_T5_model(model_name: str, tokenizer: T5Tokenizer, adapter_name: str, device: torch.device) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, output_hidden_states=True)
    model.resize_token_embeddings(len(tokenizer))

    model.add_adapter(adapter_name, config=create_pef_config(adapter_name))
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)

    model = model.to(device)

    print("Finished loading model")

    return model


def create_stuff(config: TrainingConfig, adapter_name: str):

    tokenizer = common_utils.create_tokenizer(model_name=config.model_name)

    print_gpu_utilization()
    training_elems = TrainingElements(
        create_T5_model(config.model_name, tokenizer,
                        adapter_name, config.device), tokenizer, torch.cuda.amp.GradScaler(),
        lambda model: common_utils.create_optimizer(model))
    print_gpu_utilization()

    training_data = TrainingData(config=config, tokenizer=tokenizer)

    return training_elems, training_data

def create_optimizer(model: T5ForConditionalGeneration) -> Adafactor:
    return Adafactor(
        model.parameters(),
        lr=0.1,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

def run(config: TrainingConfig) -> float:
    adapter_name = 'prefix_tuning'  # bottleneck_adapter

    training_elems, training_data = create_stuff(config, adapter_name)

    print("Training started...")
    print(f'{config.model_name=} {adapter_name} {create_pef_config(adapter_name)}\
      {config.batch_size=} {config.epochs=}')

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
                                batch_idx=batch_idx, need_to_optimize=need_to_optimize)

                losses.append(loss)

        loss = sum(losses)/len(losses)

        print(f'{loss=}')

        best_em_score = validate(training_elems, training_data, config,
                                 e, sum(losses)/len(losses), config.model_saving_folder, best_em_score)

    return best_em_score
