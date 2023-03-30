import wandb
import torch
import common_utils

from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5_finetuning_class import Finetuning


class InCtxLearning(Finetuning):
    @staticmethod
    def get_aliases():
        return ['in-context-learning']

    def get_local_aliases(self):
        return InCtxLearning.get_aliases()

    @staticmethod
    def preprocess_input(input: pd.DataFrame):
        return ("Using context: " + "'" + input['context'] + "'" + ". Answer following question: " + "'" + input['contextual_answer'] + "'" + " </s>").tolist()

    def __init__(self, config: TrainingConfig, checkpoint: str = None):

        super().__init__(config, checkpoint, closure=InCtxLearning.preprocess_input)        

    def train(self, is_wandb_sweep = False):

        # I want to try another T5 version with unlearned span corruption
        #config.model_name = 'google/t5-xxl-lm-adapt'
        self.config.eval_batch_size = self.config.batch_size

        print("Training started...")
        print(f'{self.config.model_name=} {self.config.batch_size=} {self.config.epochs=}')

        torch.cuda.empty_cache()

        best_exact_match_acc = 0.0
        for key in self.training_data.test_loaders:
            loader = self.training_data.test_loaders[key]

            exact_match_acc = evaluate(
                self.training_elems, self.config, loader, verbose=True)
            
            print(f'{key=} {exact_match_acc=}')

            if exact_match_acc > best_exact_match_acc:
                best_exact_match_acc = exact_match_acc

        return best_exact_match_acc
