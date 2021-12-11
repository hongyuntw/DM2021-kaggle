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
from utils import compute_acc , custom_class_weight
from torch.utils.data import random_split
import os
import numpy as np
import torch.nn.functional as F
from transformers import logging
logging.set_verbosity_warning()
### hyperparams ###

# pretrained_model = 'cardiffnlp/twitter-roberta-base-emotion'
pretrained_model = 'roberta-large'
lr = 1e-5

batch_size = 4
val_batch_size = 8
accumulation_steps = 12

# batch_size = 24
# val_batch_size = 48
# accumulation_steps = 2

mode = 'train'
epochs = 4
warm_up_rate = 0.05

warm_up = True
valid = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)



### hyperparams ###

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# model_name = f'twitter-roberta-base-emotion'
m = pretrained_model.split('/')[-1]
model_name  = f'{m}-hashtags-noweight'
print(model_name)
model_save_path = f'./outputs_models/{model_name}/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


log_fp = open(f'./logs/{model_name}.txt', 'w')


train_set = CrossEncoderDataset(mode, tokenizer)
print(len(train_set))
# Random split
train_set_size = int(len(train_set) * 0.9)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])
print(f'train_size : {train_set_size}, val_size {valid_set_size}')
valid_loader = DataLoader(valid_set, batch_size=val_batch_size, shuffle=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=8, ignore_mismatched_sizes=True)
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, total_steps)

# class_weight = torch.FloatTensor(custom_class_weight).to(device)
# loss_fct = nn.CrossEntropyLoss(weight=class_weight)
loss_fct = nn.CrossEntropyLoss()

model = model.to(device)
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    recall = 0.0
    f1 = 0.0
    precision = 0.0
    model.train()
    for i, data in enumerate(train_loader):        
        # input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in data]
        input_ids, attention_mask, labels = [t.to(device) for t in data]

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        # token_type_ids=token_type_ids,
                        labels=labels)
        logits = outputs.logits

        loss = loss_fct(logits, labels)
        running_loss += loss.item()
        loss = loss / accumulation_steps

        loss.backward()
        if ((i+1) % accumulation_steps) == 0 or ((i+1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        batch_acc = compute_acc(logits, labels)
        acc += batch_acc.detach().cpu()

        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )
    print(f'Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , file=log_fp)
    print('')
    # valid 
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    totals_batch = len(valid_loader)
    # val_recall = 0.0
    # val_f1 = 0.0
    # val_precision = 0.0
    for i, data in enumerate(valid_loader):        
        # input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in data[:-1]]
        input_ids, attention_mask, labels = [t.to(device) for t in data]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            # token_type_ids=token_type_ids,
                            labels=labels)
            logits = outputs.logits

            loss = loss_fct(logits, labels)

        val_loss += loss.item()
        batch_acc = compute_acc(logits, labels)
        val_acc += batch_acc

        print(f'\r[val]Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}' , end='' )
    print(f'[val]Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}' , file=log_fp )
    log_fp.flush()
    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')
