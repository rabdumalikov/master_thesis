import math
import torch
from .utils import *
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers.optimization import Adafactor, AdafactorSchedule

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import os
import re
import pandas as pd
import numpy as np
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from evaluate import load
from torch.cuda.amp import autocast
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta
from transformers.adapters import PrefixTuningConfig, AdapterConfig, CompacterConfig, LoRAConfig, IA3Config
import argparse

exact_match_metric = load("exact_match")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_data(filepath='nq_factuals_only2.csv'):
    datasets = pd.read_csv('nq_factuals_only2.csv', index_col=0)

    train_set, test_set = train_test_split(datasets, test_size=0.2, shuffle=True)

    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set

def collate_fn( input ):

    input = pd.concat(input, axis=1).T
    
    pairs = ('<question>: '+input['question']+' <context>: '+input['context']+' </s>').tolist()

    source_dict = tokenizer( pairs, # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,      # Pad & truncate all sentences.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation = True,
                        return_tensors = 'pt',     # Return pytorch tensors.
                )

    answers = (input['answer']+' </s>').tolist()
    
    target_dict = tokenizer( answers, # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'                            
                        max_length = 16,      # Pad & truncate all sentences.
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation = True,
                        return_tensors = 'pt',     # Return pytorch tensors.
                )
 
    return (source_dict['input_ids'], source_dict['attention_mask'], target_dict['input_ids'], target_dict['attention_mask'], input['question'], answers )

@torch.no_grad()
def evaluate( tokenizer, model, device, loader ):
    print("Evaluating...")

    model.eval()

    predictions = []
    actuals = []
    all_questions = []

    for batch in tqdm(loader):
        src_ids = batch[0].to(device)
        src_am  = batch[1].to(device)
        trg_ids = batch[2].to(device)
        questions = batch[4]

        target = tokenizer.batch_decode(trg_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with autocast(dtype=torch.bfloat16):        
            generated_ids = model.generate(
                input_ids = src_ids,
                attention_mask = src_am
                )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        predictions.extend(preds)
        actuals.extend(target)
        all_questions.extend(questions)

        torch.cuda.empty_cache()        

    #print(all_questions)
    #print(predictions)
    #print(actuals)

    exact_match_acc = exact_match_metric.compute(predictions=np.array( predictions ), references=np.array( actuals ) )
    
    return exact_match_acc #, accuracy_acc

def save_model( em_score, model, optimizer, loss, e, model_name, folder ):
    ext = model_name.split('-')[-1]
    print(em_score, ext, loss, e, model_name )
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'exact_match_acc': em_score
        }, f"{folder}/model_{e}_{str(loss)}_{round(em_score,3)}.pt")

def load_model(path, model, optimizer):
    # Loading checkpoint
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    em_score = checkpoint['exact_match_acc']

    return epoch, loss, em_score

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]
    
def train_step( train_batch, batch_idx ):

    torch.cuda.empty_cache()

    src_ids = train_batch[0].to(0)        
    src_am  = train_batch[1].to(0)
    trg_ids = train_batch[2].to(0)

    lm_labels = trg_ids.clone().detach()
    lm_labels[trg_ids == tokenizer.pad_token_id] = -100
    
    with autocast(dtype=torch.bfloat16):        
        loss = model(            
            input_ids=src_ids,
            attention_mask=src_am,
            labels=lm_labels.to(f'cuda:{len(available_gpus)-1}')
            )[0]

    # normalize loss to account for batch accumulation
    loss = loss / number_accumulations

    scaler.scale(loss).backward()
    
    if ((batch_idx + 1) % number_accumulations == 0) or (batch_idx + 1 == len(train_loader)):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


    if batch_idx % gpu_stat_every == 0:
        print_gpu_utilization()
        torch.cuda.empty_cache()

    return loss.item()
        
def validate( epoch, loss, folder ):
    global best_em_score

    torch.cuda.empty_cache()

    if epoch % evaluation_every != 0:
        return

    exact_match_acc = evaluate( tokenizer, model, device, test_loader )
    print( f'\te={epoch}, {exact_match_acc=}')
    
    if exact_match_acc['exact_match'] > best_em_score:
        best_em_score = exact_match_acc['exact_match']

        print(f'Saving model at e={epoch} with bestEM={best_em_score}')

        save_model( exact_match_acc['exact_match'], model, optimizer, loss, epoch, model_name, folder )

def main():

    print("Training started...")
    for e in range(1, epochs):

        model.train()
        torch.cuda.empty_cache()

        losses = []
        
        start = timer()
        for i, train_batch in enumerate(train_loader, 1):
            loss = train_step(train_batch, i)
            losses.append(loss)
        end = timer()

        print(f'loss={sum(losses)/len(losses)}')

        print(f'{e} took ', timedelta(seconds=end-start))

        validate(e, sum(losses), args.folder)

    print("\n============================\n")
    print( f'{best_em_score=}' )
                

model_name = 'google/t5-v1_1-large'

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Not sure that I would need it
SPECIAL_TOKENS_DICT = {
    'pad_token' : '<pad>',
    'additional_special_tokens' : ['<question>', '<context>']
}

tokenizer.add_special_tokens( SPECIAL_TOKENS_DICT ) 
print("Finished loading tokenizer")

device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")
print(torch.cuda.is_available())
#dev_map = {'shared': 0, 'decoder.embed_tokens': 1, 'encoder': 0, 'decoder.block.0': 1, 'decoder.block.1': 1, 'decoder.block.2': 1, 'decoder.block.3': 1, 'decoder.block.4': 1, 'decoder.block.5': 1, 'decoder.block.6': 1, 'decoder.block.7': 1, 'decoder.block.8': 1, 'decoder.block.9': 1, 'decoder.block.10': 1, 'decoder.block.11': 1, 'decoder.block.12': 1, 'decoder.block.13': 1, 'decoder.block.14': 1, 'decoder.block.15': 1, 'decoder.block.16': 1, 'decoder.block.17': 1, 'decoder.block.18': 1, 'decoder.block.19': 1, 'decoder.block.20': 1, 'decoder.block.21': 1, 'decoder.block.22': 1, 'decoder.block.23': 1, 'decoder.final_layer_norm': 1, 'decoder.dropout': 1, 'lm_head': 1}

print_gpu_utilization()

# Load the T5-XL model
# model = T5ForConditionalGeneration.from_pretrained(model_name, device_map='balanced')
# model.resize_token_embeddings( len(tokenizer) )
# model.gradient_checkpointing_enable()
# model.config.use_cache = False

adapter_name = 'prefix_tuning' #bottleneck_adapter

if adapter_name == 'prefix_tuning':
    config = PrefixTuningConfig(flat=False, prefix_length=8)
elif adapter_name == 'bottleneck_adapter':
    config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
elif adapter_name == 'compacter':
    config = CompacterConfig()
elif adapter_name == 'lora':
    config = LoRAConfig(r=8, alpha=16)
elif adapter_name == 'ia3':
    config = IA3Config()

model = T5ForConditionalGeneration.from_pretrained( model_name, output_hidden_states=True)#, device_map='balanced' )
model.resize_token_embeddings( len(tokenizer) )
model.add_adapter(adapter_name, config=config)
model.train_adapter(adapter_name)
model.set_active_adapters(adapter_name)
model = model.to(device)
#model.gradient_checkpointing_enable()
#model.config.use_cache = False

#print(model.hf_device_map)
print( "Finished loading model" )

print_gpu_utilization()

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(f'{available_gpus=}')
#for name, param in model.named_parameters():
#   print(name, param.device )


#optimizer = torch.optim.AdamW( model.parameters(), lr=0.0001)
# replace AdamW with Adafactor
optimizer = Adafactor(
    model.parameters(),
    lr=0.0001,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)

train_set, test_set = get_data()

# Create the parser
parser = argparse.ArgumentParser(description='p-tuning')

# Add the positional argument
parser.add_argument('-f', '--folder', type=str, help='folder for output')
parser.add_argument('-g', '--gpu_name', type=str, help='name of the gpu that will be used')

# Parse the arguments
args = parser.parse_args()

batch_size = 16 if args.gpu_name == '40g' else 32
number_accumulations = 2 if args.gpu_name == '40g' else 1
gpu_stat_every = 500 # training steps
evaluation_every = 1 # training steps
epochs = 100 #math.ceil(50000 / (len(train_set)//32))
best_em_score = 0.0
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

train_loader = DataLoader( PandasDataset(train_set), collate_fn=collate_fn, batch_size=batch_size, num_workers=4, pin_memory=True)
test_loader  = DataLoader( PandasDataset(test_set), collate_fn=collate_fn, batch_size=batch_size*3, num_workers=4, pin_memory=True)

print( f'{model_name=} {adapter_name} {config} {batch_size=} {epochs=}', len(train_set), len(test_set) )

if __name__ == "__main__":
    main()


