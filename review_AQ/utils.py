import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data):
    asin_dict = defaultdict(list)
    seen = defaultdict(set)

    for item in data:
        asin = item["asin"]
        
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