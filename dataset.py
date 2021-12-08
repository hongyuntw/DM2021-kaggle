import torch
from torch.utils.data import Dataset
import random
import numpy as np
import ast
import pandas as pd
from utils import emotion2idx
import ast

# for cross encoder 
class CrossEncoderDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.tok = tokenizer

        if mode == 'train':
            self.df = pd.read_csv('./train.csv')

        if mode == 'test':
            self.df = pd.read_csv('./test.csv')

    def tensorsize(self, text):
        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=256,
            return_tensors='pt',
            pad_to_max_length=True,
            padding='max_length',
            truncation='longest_first',
        )

        # print(input_dict)
        # input()
        input_ids = input_dict['input_ids'][0]
        # token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        # return (input_ids, attention_mask, token_type_ids)
        return (input_ids, attention_mask)
         
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        tweet_id = data['tweet_id']

        if self.mode == 'train':
            label = emotion2idx[data['emotion']]

        

        text = data['text']
        hash_tags =  ast.literal_eval(data['hashtags'])
        hash_tags_text = ''
        for hash_tag in hash_tags:
            hash_tags_text += ', ' + hash_tag.lower()
        # if len(hash_tags) >= 0:
            
        # print(hash_tags)
        # print(hash_tags_text)
        # print(text)
        # text = text +  hash_tags_text
        # print(text)
        # input()

        # input_ids, attention_mask, token_type_ids = self.tensorsize(text)
        input_ids, attention_mask = self.tensorsize(text)


        
        if self.mode == 'train':
            # return input_ids, attention_mask, token_type_ids, label
            return input_ids, attention_mask, label
        if self.mode == 'test':
            # return (input_ids, attention_mask, token_type_ids), (tweet_id)
            return (input_ids, attention_mask), (tweet_id)


    def __len__(self):
        return self.df.shape[0]