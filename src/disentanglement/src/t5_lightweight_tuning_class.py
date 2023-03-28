import wandb
import torch
import common_utils

from utils import *
from transformers import AdamW, get_scheduler
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import PrefixTuningConfig, AdapterConfig, LoRAConfig
from t5_finetuning_class import Finetuning

class LightweightTuning(Finetuning):

    @staticmethod
    def get_aliases():
        return ['prefix_tuning', 'bottleneck_adapter', 'lora']

    def get_local_aliases(self):
        return LightweightTuning.get_aliases()

    def __init__(self, config: TrainingConfig, checkpoint: str = None):
        # NOTE: LightweightTuning will break if it would be True
        config.gradient_checkpointing_enable = False
        super().__init__(config, checkpoint)

    def create_T5_model(self, checkpoint: str) -> T5ForConditionalGeneration:
        
        model_name = checkpoint if checkpoint else self.config.model_name

        model = T5ForConditionalGeneration.from_pretrained(
            model_name, output_hidden_states=True)
    
        if checkpoint is None:
            model.add_adapter(self.config.tuning_method, config=self.create_pef_config())
            model.train_adapter(self.config.tuning_method)
        
        model.set_active_adapters(self.config.tuning_method)

        print("Finished loading model")

        return model

    def create_pef_config(self):

        adapter_name = self.config.tuning_method

        if adapter_name == 'prefix_tuning':
            return PrefixTuningConfig(flat=True, prefix_length=100) # based on prefix-tuning paper
        elif adapter_name == 'bottleneck_adapter':
            return AdapterConfig(mh_adapter=True, output_adapter=True,
                                reduction_factor=16, non_linearity="relu")
        elif adapter_name == 'lora':
            return LoRAConfig(r=16, alpha=32) # based on adapters paper
        else:
            return None
