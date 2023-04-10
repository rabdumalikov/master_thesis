import wandb
import torch
import common_utils

from utils import *
from transformers import AdamW, get_scheduler
from transformers.optimization import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5_finetuning_class import Finetuning
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, LoraConfig, PromptTuningConfig

class PEFTTuning(Finetuning):

    @staticmethod
    def get_aliases():
        return ['peft_prefix_tuning', 'peft_prompt_tuning', 'peft_lora']

    def get_local_aliases(self):
        return PEFTTuning.get_aliases()

    def __init__(self, config: TrainingConfig, checkpoint: str = None):
        # NOTE: PEFTTuning might break if it would be True
        config.gradient_checkpointing_enable = False
        super().__init__(config, checkpoint)

    def create_T5_model(self, checkpoint: str) -> T5ForConditionalGeneration:
        
        if checkpoint is None:
            model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name, output_hidden_states=True)

            peft_config = self.create_pef_config()
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            config = PeftConfig.from_pretrained(checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, output_hidden_states=True)
            model = PeftModel.from_pretrained(model, checkpoint)
        
        print("Finished loading model")

        return model

    def create_pef_config(self):

        adapter_name = self.config.tuning_method
        
        if adapter_name == 'peft_prefix_tuning':
            return PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, 
                num_virtual_tokens=self.config.tuning_settings['prefix_length'])
        elif adapter_name == 'peft_prompt_tuning':
            return PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, 
                num_virtual_tokens=self.config.tuning_settings['prompt_length'])
        elif adapter_name == 'peft_lora':
            return LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.config.tuning_settings['lora_r'], 
                lora_alpha=self.config.tuning_settings['lora_alpha'], lora_dropout=0.1)
        else:
            return None
