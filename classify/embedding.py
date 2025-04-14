import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
import json

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print(f"Error decoding JSON: {line}")
                continue
    return data    

def get_embeddings(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :] .numpy()

data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/McMarket/'
country1 = ['br', 'cn', 'fr', 'jp', 'mx']
country2 = ['au', 'ca', 'de', 'es', 'in', 'it', 'nl', 'sa']

review_data = []

# read data
for c in country1:
    data = read_jsonl(data_path + f'{c}_reviews_translated.jsonl')
    review_data.extend([[i['translatedReview'], c] for i in data])

for c in country2:
    data = read_jsonl(data_path + f'{c}_reviews.jsonl')
    review_data.extend([[i['reviewText'], c] for i in data])

# convert to DataFrame
df = pd.DataFrame(review_data, columns=['text', 'label'])

# get embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df['label'].unique()))
model.eval()

df['embeddings'] = get_embeddings(model, tokenizer, df['text'].tolist())
df.to_csv('embeddings.csv', index=False)

