import torch
from transformers import AdamW, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from dataset import CrossEncoderDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import compute_acc , custom_class_weight, idx2emotion
from torch.utils.data import random_split
import os
import numpy as np
import torch.nn.functional as F
from transformers import logging
import pandas as pd
logging.set_verbosity_warning()
### hyperparams ###

# pretrained_model = 'cardiffnlp/twitter-roberta-base-emotion'
pretrained_model = 'roberta-large'
batch_size = 256
mode = 'test'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)



### hyperparams ###

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

model_name = pretrained_model.split('/')[-1]
model_name = 'roberta-large-hashtags-noweight'
print(model_name)
model_save_path = f'./outputs_models/{model_name}/model_3.pt'
print(model_save_path)

test_set = CrossEncoderDataset(mode, tokenizer)
print(len(test_set))

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=8, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(model_save_path), strict=False)
model = model.to(device)
model.eval()

all_tweet_ids = []
all_preds = []

with torch.no_grad():
    totals_batch = len(test_loader)
    for i, data in enumerate(test_loader):        
        input_ids, attention_mask = [t.to(device) for t in data[0]]

        tweet_ids = data[1]
        all_tweet_ids += list(tweet_ids)
        
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask)
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        all_preds += list(preds)
        print(f'\r batch : {i+1}/{totals_batch}' , end='' )


all_emotions = []
for pred in all_preds:
    all_emotions.append(idx2emotion[pred])
    
df = pd.DataFrame(
    {'id': all_tweet_ids,
     'emotion': all_emotions,
    })

df.to_csv(f'./{model_name}.csv', index=False)
    
