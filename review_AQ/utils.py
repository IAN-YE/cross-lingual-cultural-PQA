import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict
import evaluate

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data, type='translate'):
    asin_dict = defaultdict(list)
    seen = defaultdict(set)

    for item in data:
        asin = item["asin"]

        if type == 'translate':
            try:
                review = item["translatedReview"]
            except:
                review = item["reviewText"]
        else:
            review = item["reviewText"]
            
        if review not in seen[asin]:
            seen[asin].add(review)
            asin_dict[asin].append(item)

    return asin_dict

def split_dataset(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=seed)

    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_size), random_state=seed)

    return train_data, val_data, test_data

def bleu_score(hypothesis, reference):
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
    return bleu_score['bleu']

def rouge_score(hypothesis, reference):
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=hypothesis, references=reference)
    return rouge_score['rougeL']

def bert_score(hypothesis, reference):
    bert = evaluate.load("bertscore", module_type="metric")
    results = bert.compute(predictions=hypothesis, references=reference, model_type="distilbert-base-uncased", lang="en")
    return sum(results['f1'])/len(results['f1'])