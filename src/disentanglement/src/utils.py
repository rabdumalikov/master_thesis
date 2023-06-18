import re
import os
import gzip
import math
import wandb
import torch
import tarfile
import numpy as np
import pandas as pd
import evaluation_script_squad_v2 as ev

from pynvml import *
from tqdm import tqdm
from evaluate import load
from bertviz import head_view
from datetime import timedelta
from timeit import default_timer as timer
from typing import Tuple, List, Dict, Optional
from torch.cuda.amp import autocast, GradScaler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, RandomSampler

class TimeMeasure:
    def __init__(self, epoch: int, steps: int):
        self.start = ''
        self.epoch = epoch
        self.steps = steps

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, type, value, traceback):
        elapsed_time = timedelta(seconds=timer()-self.start)

        print(f'Epoch={self.epoch} took {elapsed_time}')

        duration_table = wandb.Table(columns=["Epoch", "Duration(in hours)"], 
            data=[[self.epoch,  elapsed_time.total_seconds()/3600]])

        bar = wandb.plot.bar(duration_table, 'Epoch', "Duration(in hours)", title=f'Epoch duration in hours')
        wandb.log({f"Duration_bar": bar})

        wandb.log({'epoch': self.epoch, 'steps': self.steps, 'epoch_duration(m)': (
            elapsed_time.total_seconds()/3600)})


class TrainingConfig:
    def __init__(self, model_name: str, num_gpus: int,
                 gradient_accumulation_steps: int,
                 batch_size: int,
                 gpu_stat_every: int, evaluation_every: int, device: torch.device,
                 dataset_type: str,
                 epochs: int,
                 model_saving_folder: str,
                 tuning_method: str,
                 gpu_name: str,
                 skip_train: bool,
                 val_accuracy: bool,
                 FP16: bool = False ):

        self.model_name = model_name
        self.num_gpus = num_gpus
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.gpu_stat_every = gpu_stat_every
        self.evaluation_every = evaluation_every
        self.eval_batch_size = batch_size * 1
        self.device = device
        self.dataset_type = dataset_type
        self.epochs = epochs
        self.FP16 = FP16
        self.model_saving_folder = model_saving_folder
        self.closure_to_save_model = save_model
        self.tuning_method = tuning_method
        self.max_length = 80
        self.repetition_penalty = 2.5
        self.length_penalty = 1.0
        self.early_stopping = True
        self.use_cache = True
        self.gradient_checkpointing_enable = True
        self.data_usage_percentage = 1.0
        self.is_wandb_sweep = False
        self.gpu_name = gpu_name
        self.skip_train = skip_train
        self.val_accuracy = val_accuracy

        self.tuning_settings = {'learning_rate': 0.0001, 
            'type_of_optimizer': 'adafactor',
            'prompt_length': 100,
            'prefix_length': 100, 
            'lora_r': 8, 
            'lora_alpha': 32, 
            'adapter_reduction_factor': 16 }


class TrainingData:

    def __init__(self, config: TrainingConfig,
                 allowed_test_sets: List[int] = ['f', 'cf', 'a(e)', 'a(r)'], **kwargs):

        test_mapping = {'factual': 'f', 'counterfactual': 'cf',
                        'closed_book': 'a(e)', 'random_context': 'a(r)'}

        train_set, val_set, test_set = get_data(
            config.dataset_type, test_mapping)

        # shuffle the DataFrame rows
        train_set = train_set.sample(frac = config.data_usage_percentage)
        val_set = val_set.sample(frac = config.data_usage_percentage)

        #train_set = train_set[:int(len(train_set)*config.data_usage_percentage)]
        #val_set = val_set[:int(len(val_set)*config.data_usage_percentage)]


        # updating epoch
        # new_num_epochs = math.ceil(50000/(len(train_set)//config.batch_size))
        # print(f'Epochs changed from {config.epochs} to {new_num_epochs}')
        # config.epochs = new_num_epochs

        print(
            f'TrainingData: {len(train_set)=} {len(val_set)=} {len(test_set)=}')

        self.train_loader = DataLoader(PandasDataset(train_set), collate_fn=lambda inp: collate_fn(
            inp, max_source_input_len=256, **kwargs), batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(PandasDataset(val_set), collate_fn=lambda inp: collate_fn(
            inp, max_source_input_len=256, **kwargs), batch_size=config.eval_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.test_loaders = {}
        for k in test_set:
            if k not in allowed_test_sets:
                continue
            
            if len(test_set[k]) == 0:
                continue

            self.test_loaders[k] = DataLoader(PandasDataset(test_set[k]), collate_fn=lambda inp: collate_fn(
                inp, max_source_input_len=396, **kwargs), batch_size=config.eval_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def to_readable_name(self, abbreviation: str):
        if abbreviation == 'f': 
            return "Factual"
        elif abbreviation == 'cf': 
            return "Counterfactual"
        elif abbreviation == 'a(e)': 
            return "Empty"
        elif abbreviation == 'a(r)': 
            return "Random"


class TrainingElements:
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, optimizer, prompt_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer, self.scheduler = optimizer
        self.prompt_model = prompt_model


class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def get_dataframe(self):
        return self.dataframe
        
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
            output.append(self.dic[k][index])
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


def get_data(dataset_type: str, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_filepath, val_filepath, test_filepath = _get_data_path_for(
        dataset_type)

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

    #return f'google/t5-v1_1-{short_name}'

    #return f'google/t5-{short_name}-lm-adapt'

    mapping = {'large': 'large', 'xxl': '11b', 'xl': '3b'}
    return f't5-{mapping[short_name]}'

    #return f'google/flan-t5-{short_name}'
    #return f'google/t5-v1_1-{short_name}'
    # names = {
    #     'large': 'google/t5-v1_1-large',
    #     'xl': 'google/t5-v1_1-xl',
    #     'xxl': 'google/t5-v1_1-xxl',
    # }

    # return names[short_name]

# experiment id(or tuning method) correspond to choosen method like finetuning, adapters, lora, etc.
# dataset id correspond to choosen dataset to train and test agaings like 'f', 'f+cf', 'f+a', etc.

def get_number_training_steps( epoch: int, num_examples: int, batch_size: int ) -> int:
    return (epoch * num_examples) // batch_size

def _get_data_path_for(dataset_type: str) -> Tuple[str, str, str]:

    director = '../their_data/'
    Experiments = {
        'cb': ('cb_train.csv.tar.gz', 'cb_val.csv.tar.gz', 'test_sets.csv.tar.gz'),        
        's(a2)': ('unanswerable_train_set.csv.tar.gz', 'GPT_f_jailbreak_irrelavent_answ_and_ctx.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f)': ('(s) f - train.csv.tar.gz', '(s) f - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f+a2)': ('new_train_set.csv.tar.gz', '(s) f - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f+a3)': ('squad_unans_train_set.csv.tar.gz', '(s) f - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f+a4)': ('squad_only_unans_train_set.csv.tar.gz', '(s) f - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f+cf)': ('(s) f+cf - train.csv.tar.gz', '(s) f+cf - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f+a)': ('(s) f+a - train.csv.tar.gz', '(s) f+a - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        's(f+cf+a)': ('(s) f+cf+a - train.csv.tar.gz', '(s) f+cf+a - val.csv.tar.gz', 'test_sets.csv.tar.gz'),
        'gpt_rnd': ('(s) f+cf - train.csv.tar.gz', '(s) f+cf - val.csv.tar.gz', 'GPT_rnd.csv.tar.gz'), #GPT_rnd.csv.tar.gz GPT_pert_ctx.csv.tar.gz
        'gpt_cf': ('(s) f+cf - train.csv.tar.gz', '(s) f+cf - val.csv.tar.gz', 'GPT_cf.csv.tar.gz'), #GPT_rnd.csv.tar.gz GPT_pert_ctx.csv.tar.gz
        'gpt_perm': ('(s) f+cf - train.csv.tar.gz', '(s) f+cf - val.csv.tar.gz', 'GPT_pert_ctx.csv.tar.gz'), #GPT_rnd.csv.tar.gz GPT_pert_ctx.csv.tar.gz
        'hmo': ('(s) f - train.csv.tar.gz', 'GPT_rnd.csv.tar.gz', 'GPT_pert_ctx.csv.tar.gz'), #(s) f - train.csv.tar.gz,
        'gpt_rnd_2': ('(s) f - train.csv.tar.gz', '(s) f - val.csv.tar.gz', 'GPT_relavent_answ_and_ctx.csv.tar.gz'),
        'gpt_rnd_2_cf_jb': ('(s) f - train.csv.tar.gz', '(s) f - val.csv.tar.gz', 'GPT_cf_jailbreak_irrelavent_answ_and_ctx.csv.tar.gz'), #GPT_cf_jailbreak_irrelavent_answ_and_ctx
        'gpt_rnd_2_f_jb': ('(s) f - train.csv.tar.gz', '(s) f - val.csv.tar.gz', 'GPT_f_jailbreak_irrelavent_answ_and_ctx.csv.tar.gz'),
        'gpt_rnd_2_f_subs_jb': ('(s) f - train.csv.tar.gz', '(s) f - val.csv.tar.gz', 'GPT_f_SUBS_jailbreak_irrelavent_answ_and_ctx.csv.tar.gz') 
    }

    train, val, test = Experiments[dataset_type]

    return director+train, director+val, director+test


def get_DisentQA_results(dataset_type: str) -> Optional[float]:

    DisentQAResults = {
        's(f)':     {'f': 76.34, 'cf': 67.84},
        's(f+cf)':  {'f': 75.75, 'cf': 76.04},
    }

    return DisentQAResults[dataset_type] if dataset_type in DisentQAResults else None


def collate_fn(input: pd.DataFrame, max_source_input_len: int, tokenizer: T5Tokenizer, closure=None, postprocessing=None):

    input = pd.concat(input, axis=1).T

    # for in-context-learning
    if closure is not None:
        que_ctx = closure(input)
    else:
        que_ctx = input['input'].values.tolist()
        #que_ctx = ("question: " + input['question'] + "\ncontext:").tolist()

    source_dict = tokenizer(que_ctx,  # Sentence to encode.
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            # Pad & truncate all sentences.
                            max_length=max_source_input_len,
                            padding='max_length',
                            # Construct attn. masks.
                            return_attention_mask=True,
                            truncation=True,
                            return_tensors='pt',     # Return pytorch tensors.
                            )

    # for prompt-tuning
    if postprocessing is not None:
        source_dict = postprocessing(source_dict)

    answers = input["contextual_answer"].values.tolist()

    target_dict = tokenizer(answers,  # Sentence to encode.
                            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                            max_length=32,      # Pad & truncate all sentences.
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

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

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

    predictions_old = []
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

        def generate():
            return training_elements.model.generate(
                input_ids=src_ids,
                attention_mask=src_am,
                max_length=config.max_length,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping, 
                use_cache=config.use_cache,
                **kwargs
            )

        if config.gpu_name == 'tesla':
            generated_ids = generate()
        else:
            with autocast(dtype=torch.bfloat16, enabled=config.FP16):
                generated_ids = generate()

        preds = training_elements.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        predictions.extend( [ ev.normalize_answer(p) for p in preds ] )
        ground_truth.extend( [ ev.normalize_answer(t) for t in target ] )
        
        all_questions.extend(questions)

        torch.cuda.empty_cache()

    if verbose:
        print(all_questions)
        print(predictions)
        print(ground_truth)

    exact_match_acc = exact_match_metric.compute(
        predictions=np.array(predictions), references=np.array(ground_truth))

    return exact_match_acc['exact_match'], lambda current, prev: current > prev

@torch.no_grad()
def validation(training_elements: TrainingElements, config: TrainingConfig, data_loader: DataLoader, verbose, **kwargs):

    print("Validating...")

    print_gpu_utilization()

    training_elements.model.eval()

    torch.cuda.empty_cache()

    losses = []
    for val_batch in tqdm(data_loader):
        def step():
            src_ids, src_am, lm_labels = unroll_batch( val_batch, 
                config.device, training_elements.tokenizer.pad_token_id )

            tokens = training_elements.tokenizer.convert_ids_to_tokens(src_ids[0]) 
            print(f'{tokens=}')
            src_ids = torch.unsqueeze( src_ids[0][:len(tokens)], dim=0 )
            src_am = torch.unsqueeze( src_am[0][:len(tokens)], dim=0 )
            lm_labels = torch.unsqueeze( lm_labels[0][:len(tokens)], dim=0 )

            def get_loss():
                return training_elements.model(
                    input_ids=src_ids,
                    attention_mask=src_am,
                    labels=lm_labels.to(f'cuda:{config.num_gpus-1}'),
                    **kwargs
                    )[0]

            if config.gpu_name == 'tesla':
                loss = get_loss()
            else:
                with autocast(dtype=torch.bfloat16, enabled=config.FP16):
                    loss = get_loss()
            # Returning only loss value important, otherwise OOM error
            return loss.item()
        
        loss = step()
        losses.append(loss)
        
    return np.mean(losses), lambda current, prev: current > prev

def validate(training_elements: TrainingElements, training_data: TrainingData,
             training_config: TrainingConfig, current_epoch: int, loss: float,
             folder: str, best_val_loss: float, verbose: bool = False, **kwargs):

    torch.cuda.empty_cache()

    if current_epoch % training_config.evaluation_every != 0:
        return

    param_name = ''
    if training_config.val_accuracy:
        val_loss, cmp = evaluate(
                training_elements, training_config, training_data.val_loader, verbose, **kwargs)
        param_name = 'val_EM_acc'
    else:
        val_loss, cmp = validation(
                training_elements, training_config, training_data.val_loader, verbose, **kwargs)
        param_name = 'val_loss'

    print(f'\te={current_epoch}, {param_name}={val_loss}')

    wandb.log({'epoch': current_epoch,
                'loss': loss, f'{param_name}': val_loss})

    if training_config.is_wandb_sweep:
        print("Skipped evaluation on the TEST set")
        return val_loss if cmp(val_loss, best_val_loss) else best_val_loss

    if cmp(val_loss, best_val_loss):

        print(f'Saving model at e={current_epoch}')

        training_config.closure_to_save_model(training_elements, val_loss, loss,
                    current_epoch, training_config.model_name, folder)

        results = {}
        for key in training_data.test_loaders:

            loader = training_data.test_loaders[key]

            exact_match_acc, _ = evaluate(
                training_elements, training_config, loader, verbose, **kwargs)

            results[training_data.to_readable_name(key)] = exact_match_acc

            wandb.log({'epoch': current_epoch,
                    'loss': loss, f'{key}_EM_acc': exact_match_acc})

            print(f'\t{key=} e={current_epoch}, {exact_match_acc=}')

        results_table = wandb.Table(columns=["Method(Dataset)", "Factual", "Counterfactual", "Empty", "Random"], 
            data=[[f'{training_config.tuning_method}({training_config.dataset_type})', 
                results['Factual'], results['Counterfactual'], results['Empty'], results['Random']]])

        for col in results_table.columns[1:]:
            bar = wandb.plot.bar(results_table, 'Method(Dataset)', col, title=f'EM accuracy on {col}')
            wandb.log({f"{col}_bar": bar})

        wandb.log({"results_table": results_table})

    return val_loss if cmp(val_loss, best_val_loss) else best_val_loss

def deepspeed_train_step(training_elements: TrainingElements, config: TrainingConfig,
               train_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

    torch.cuda.empty_cache()

    src_ids, src_am, lm_labels = unroll_batch( train_batch, 
        config.device, training_elements.tokenizer.pad_token_id )

    def get_loss():
        return training_elements.model(
            input_ids=src_ids,
            attention_mask=src_am,
            labels=lm_labels.to(f'cuda:{config.num_gpus-1}'),
            **kwargs
            )[0]

    loss = get_loss()
    training_elements.model.backward(loss)
    training_elements.model.step()

    # This is really important '.item()' because otherwise pytorch accumulates memory losses,
    # and I get OOM error.
    return loss.item()


def train_step(training_elements: TrainingElements, config: TrainingConfig,
               train_batch, batch_idx: int, need_to_optimize: bool, **kwargs):

    torch.cuda.empty_cache()

    src_ids, src_am, lm_labels = unroll_batch( train_batch, 
        config.device, training_elements.tokenizer.pad_token_id )

    def get_loss():
        return training_elements.model(
            input_ids=src_ids,
            attention_mask=src_am,
            labels=lm_labels.to(f'cuda:{config.num_gpus-1}'),
            **kwargs
            )[0]

    if config.gpu_name == 'tesla':
        loss = get_loss()
    else:
        with autocast(dtype=torch.bfloat16, enabled=config.FP16):
            loss = get_loss()

    # normalize loss to account for batch accumulation
    loss = loss / config.gradient_accumulation_steps

    training_elements.scaler.scale(loss).backward()

    if need_to_optimize:
        training_elements.scaler.step(training_elements.optimizer)
        training_elements.scaler.update()
        
        if training_elements.scheduler is not None:
            training_elements.scheduler.step() 

        training_elements.optimizer.zero_grad()

    if batch_idx % config.gpu_stat_every == 0:
        print_gpu_utilization()
        torch.cuda.empty_cache()

    # This is really important '.item()' because otherwise pytorch accumulates memory losses,
    # and I get OOM error.
    return loss.item()

def unroll_batch( batch, device: torch.device, pad_token_id ):
    input_ids = batch[0]#.to(device)
    attention_mast = batch[1]#.to(device)
    target_ids = batch[2]#.to(device)

    labels = target_ids.clone().detach()
    labels[target_ids == pad_token_id] = -100


    return input_ids, attention_mast, labels

def deduce_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device

def get_dataset_name_choices() -> List[str]:
    return ['s(f)', 's(f+a2)', 's(f+a3)', 's(f+a4)', 's(f+cf)', 's(f+a)', 's(f+cf+a)', 'gpt_rnd', 'gpt_rnd_2', 'gpt_rnd_2_cf_jb', 'gpt_rnd_2_f_jb', 'hmo', 'gpt_perm', 'cb', 'gpt_cf', 's(a2)', 'gpt_rnd_2_f_subs_jb']

def get_model_name_choices() -> List[str]:
    return ['large', 'xl', 'xxl', 'small', 'base']

def get_tuning_type(id: int):
    build_file = f'.builds/{id}/build_{id}.sh'
    with open(build_file, 'r') as f:
        #python -u main_class.py --dataset_type 's(f+a)' -t promptuning -b 32 --grad_accum 1 -m large -s .builds/568/models/ -g tesla -p 568

        for l in f.readlines():
            pattern = re.compile(r'^python.* (.*)\.py.* -t (.*?) .*$')
            for match in pattern.finditer(l):
                return match.group(2)

    return 'Such file doesnt exist!'

def find_best_checkpoint(id: int):
    dir_path = f'.builds/{id}/models'

    checkpoint_name = ''
    best_em = -1
    # iterate over all the directories in the specified directory
    for dir_name in os.listdir(dir_path):

        if os.path.isdir(os.path.join(dir_path, dir_name)):

            with open(dir_path + '/' + dir_name  + '/' + 'results.txt', 'r') as f:
                d = {}
                for l in f.readlines():
                    key, value = l.split('=')
                    d[key] = value
                    
                if float(d['em']) >= best_em:
                    best_em = float(d['em'])
                    checkpoint_name = dir_name

                    print(f'Best checkpoint {checkpoint_name} with em={best_em}')

    return dir_path + '/' + checkpoint_name

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)