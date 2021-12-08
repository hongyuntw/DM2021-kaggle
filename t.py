from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

import twikenizer as twk
twk = twk.Twikenizer()

task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
# MODEL = 'roberta-base'
# MODEL = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print(tokenizer.unk_token_id)

text = 'i just love it when every single one of my songs just delete themselves..😡😒 this is the 3rd times this has happened! #notamused'

# text = preprocess(text)
# encoded_input = tokenizer(text, return_tensors='pt')

tokens = tokenizer.tokenize(text)
twk_tokens = twk.tokenize(text)
print(twk_tokens)
print(tokens)

print('-------')
ids = tokenizer.convert_tokens_to_ids(tokens)
twk_ids = tokenizer.convert_tokens_to_ids(twk_tokens)
print(ids)
print(twk_ids)

print('-------')

t = tokenizer.convert_ids_to_tokens(ids)
twk_t = tokenizer.convert_ids_to_tokens(twk_ids)

print(t)
print(twk_t)

# print(encoded_input)