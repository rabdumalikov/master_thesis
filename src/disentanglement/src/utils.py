import os
import gzip
import wandb
import torch
import tarfile
import numpy as np
import pandas as pd


from pynvml import *
from tqdm import tqdm
from evaluate import load
from datetime import timedelta
from typing import Tuple, List, Dict
from timeit import default_timer as timer
from torch.cuda.amp import autocast, GradScaler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

class TimeMeasure:
    def __init__(self, epoch: int):
        self.start = ''
        self.epoch = epoch

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, type, value, traceback):
        elapsed_time = timedelta(seconds=timer()-self.start)

        print(f'Epoch={self.epoch} took {elapsed_time}')

        wandb.log({'epoch': self.epoch, 'elapsed_time': elapsed_time.total_seconds()})

class TrainingConfig:
    def __init__(self, model_name: str, num_gpus: int,
                 gradient_accumulation_steps: int,
                 batch_size: int,
                 gpu_stat_every: int, evaluation_every: int, device: torch.device,
                 experiment_id: int,
                 epochs: int,
                 model_saving_folder: str,
                 FP16: bool = False):

        self.model_name = model_name
        self.num_gpus = num_gpus
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.gpu_stat_every = gpu_stat_every
        self.evaluation_every = evaluation_every
        self.eval_batch_size = batch_size * 3
        self.device = device
        self.experiment_id = experiment_id
        self.epochs = epochs
        self.FP16 = FP16
        self.model_saving_folder = model_saving_folder


class TrainingData:
    def __init__(self, config: TrainingConfig, tokenizer: T5Tokenizer, closure=None, 
                 allowed_test_sets: List[int] = ['f', 'cf', 'a(e)', 'a(r)']):

        test_mapping = {'factual': 'f', 'counterfactual': 'cf', 
            'closed_book': 'a(e)', 'random_context': 'a(r)' }

        train_set, val_set, test_set = get_data(config.experiment_id, test_mapping)

        print(
            f'TrainingData: {len(train_set)=} {len(val_set)=} {len(test_set)=}')

        self.train_loader = DataLoader(PandasDataset(train_set), collate_fn=lambda inp: collate_fn(
            inp, tokenizer, closure), batch_size=config.batch_size, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(PandasDataset(val_set), collate_fn=lambda inp: collate_fn(
            inp, tokenizer, closure), batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)

        self.test_loaders = {}
        for k in test_set:
            if k not in allowed_test_sets:
                continue

            self.test_loaders[k] = DataLoader(PandasDataset(test_set[k]), collate_fn=lambda inp: collate_fn(
                inp, tokenizer, closure), batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)

class TrainingElements:
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, scaler: GradScaler, optimizer, prompt_model = None):
        self.model = model
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.optimizer = optimizer( prompt_model if prompt_model is not None else model)
        self.prompt_model = prompt_model

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

class DictDataset(Dataset):
    def __init__(self, dic):
        self.dic = dic

    def __len__(self):
        return len(self.dic[self.dic.keys()[0]])

    def __getitem__(self, index):
        output = []
        for k in self.dic:
            output.append( self.dic[k][index] )
        return output

def get_number_of_epochs(epochs: int) -> int:
    return epochs  # math.ceil(50000 / (len(train_set)//32))


def get_number_of_gpus() -> int:
    available_gpus = [torch.cuda.device(i)
                      for i in range(torch.cuda.device_count())]
    print(f'{available_gpus=}')

    return len(available_gpus)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    # [ torch.cuda.memory_allocated(f'cuda:{c}')*0.000001 for c in range(2) ]
    mallocs = []

    print(f"\t GPU memory occupied: {info.used//1024**2} MB. {mallocs=} MB")


def _read_tar_gz_context(filepath: str) -> pd.DataFrame:

    with tarfile.open(filepath, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.csv'):
                csv_file = tar.extractfile(member)
                break

        return pd.read_csv(csv_file)


def get_data(experiment_id: int, mapping: Dict[str,str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_filepath, val_filepath, test_filepath = _get_data_path_for(
        experiment_id)

    # obtaining train data
    train_set = _read_tar_gz_context(train_filepath)

    # obtaining validation data
    val_set = _read_tar_gz_context(val_filepath)

    # obtaining test data
    test_set = _read_tar_gz_context(test_filepath)

    test_sets = {}
    for k in mapping:
        k_test = test_set[test_set['type'] == k]
        k_test.reset_index(drop=True, inplace=True)
        test_sets[mapping[k]] = k_test
    
    return train_set, val_set, test_sets


def get_model_name(short_name: str) -> str:

    names = {
        'large': 'google/t5-v1_1-large',
        'xl': 'google/t5-v1_1-xl',
        'xxl': 'google/t5-v1_1-xxl',
    }

    return names[short_name]


def _get_data_path_for(experiment_id: int) -> Tuple[str, str, str]:

    director = '../their_data/'
    Experiments = {
        'exp1': ('(s) f - train.csv.tar.gz', '(s) f - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        'exp2': ('(s) f+cf - train.csv.tar.gz', '(s) f+cf - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        'exp3': ('(s) f+a - train.csv.tar.gz', '(s) f+a - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        'exp4': ('(s) f+cf+a - train.csv.tar.gz', '(s) f+cf+a - val.csv.tar.gz', 'test_sets.csv.tar.gz')
    }

    train, val, test = Experiments[f'exp{experiment_id}']

    return director+train, director+val, director+test


def collate_fn(input: pd.DataFrame, tokenizer: T5Tokenizer, closure=None):

    input = pd.concat(input, axis=1).T

    if closure is not None:
        que_ctx = closure(input)
    else:
        que_ctx = input['input'].values.tolist()

    source_dict = tokenizer(que_ctx,  # Sentence to encode.
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            # Pad & truncate all sentences.
                            max_length=512,
                            padding='max_length',
                            # Construct attn. masks.
                            return_attention_mask=True,
                            truncation=True,
                            return_tensors='pt',     # Return pytorch tensors.
                            )

    answers = input['contextual_answer'].values.tolist()

    target_dict = tokenizer(answers,  # Sentence to encode.
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            max_length=16,      # Pad & truncate all sentences.
                            padding='max_length',
                            # Construct attn. masks.
                            return_attention_mask=True,
                            truncation=True,
                            return_tensors='pt',     # Return pytorch tensors.
                            )

    return (source_dict['input_ids'], source_dict['attention_mask'], target_dict['input_ids'], target_dict['attention_mask'], input['question'].values.tolist())


def save_model(training_elements: TrainingElements, em_score: float, loss: float, e: int, model_name: str, folder: str):
    ext = model_name.split('-')[-1]

    print(em_score, ext, loss, e, model_name)

    checkpoint_path = f'{folder}/checkpoint_{e}'
    
    os.mkdir(checkpoint_path)

    with open(f'{checkpoint_path}/results.txt', 'w') as f:
        f.write(f'epoch={e}\nloss={loss}\nem={round(em_score,3)}')

    training_elements.model.save_pretrained(
        f"{checkpoint_path}/")
    training_elements.tokenizer.save_pretrained(
        f"{checkpoint_path}/")

@torch.no_grad()
def evaluate(training_elements: TrainingElements, config: TrainingConfig,
             data_loader: DataLoader,
             verbose: bool = False, **kwargs) -> float:

    print("Evaluating...")

    training_elements.model.eval()

    exact_match_metric = load("exact_match")

    predictions = []
    ground_truth = []
    all_questions = []

    for batch in tqdm(data_loader):
        src_ids = batch[0].to(config.device)
        src_am = batch[1].to(config.device)
        trg_ids = batch[2].to(config.device)
        questions = batch[4]

        target = training_elements.tokenizer.batch_decode(
            trg_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        kwargs = {}
        if training_elements.prompt_model is not None:
            prompt = training_elements.prompt_model(batch_size=batch[0].size(
                    0), device=config.device)
            kwargs['prompt'] = prompt

        with autocast(dtype=torch.bfloat16, enabled=config.FP16):
            generated_ids = training_elements.model.generate(
                input_ids=src_ids,
                attention_mask=src_am,
                **kwargs
            )

        preds = training_elements.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        predictions.extend(preds)
        ground_truth.extend(target)
        all_questions.extend(questions)

        torch.cuda.empty_cache()

    if verbose:
        print(all_questions)
        print(predictions)
        print(ground_truth)

    exact_match_acc = exact_match_metric.compute(
        predictions=np.array(predictions), references=np.array(ground_truth))

    return exact_match_acc['exact_match']


def validate(training_elements: TrainingElements, training_data: TrainingData,
             training_config: TrainingConfig, current_epoch: int, loss: float, folder: str, best_em_score: float, verbose: bool = False, **kwargs):

    torch.cuda.empty_cache()

    if current_epoch % training_config.evaluation_every != 0:
        return

    for key in training_data.test_loaders:
        loader = training_data.test_loaders[key]

        exact_match_acc = evaluate(
            training_elements, training_config, loader, verbose, **kwargs)

        wandb.log({'epoch': current_epoch,
                'loss': loss, f'{key}_EM_acc': exact_match_acc})

        print(f'\t{key=} e={current_epoch}, {exact_match_acc=}')

        if exact_match_acc > best_em_score:
            best_em_score = exact_match_acc

            print(
                f'Saving {key} model at e={current_epoch} with bestEM={best_em_score}')

            save_model(training_elements, exact_match_acc, loss,
                    current_epoch, training_config.model_name, folder)

    return best_em_score


def train_step(training_elements: TrainingElements, config: TrainingConfig,
               train_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

    torch.cuda.empty_cache()

    src_ids = train_batch[0].to(0)
    src_am = train_batch[1].to(0)
    trg_ids = train_batch[2].to(0)

    lm_labels = trg_ids.clone().detach()
    lm_labels[trg_ids == training_elements.tokenizer.pad_token_id] = -100

    with autocast(dtype=torch.bfloat16, enabled=config.FP16):
        loss = training_elements.model(
            input_ids=src_ids,
            attention_mask=src_am,
            labels=lm_labels.to(f'cuda:{config.num_gpus-1}'),
            **kwargs
        )[0]

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


def deduce_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device
