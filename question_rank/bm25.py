from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json

class BM25:
    def __init__(self, corpus):
        corpus = [doc.lower() for doc in corpus]
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_top_n(self, query, n=5):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        print(doc_scores)
        top_n = [self.tokenized_corpus[i] for i in np.argsort(doc_scores)[::-1][:n]]
        return top_n

def split_dataset(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=seed)

    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_size), random_state=seed)

    return train_data, val_data, test_data

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def single_market(data):
    corpus = []
    for i in data:
        corpus.append(i['question'])
    return corpus

if __name__ == '__main__':
    country2 = ['au','ca','uk','in']
    country1 = ['br','cn','fr','jp','mx']

    country = ['au']
    
    data_path = '../../../data_PQA/McMarket/McMarket_all/'

    for c in country:
        print(c)
        data = read_jsonl(data_path + f'McMarket_LLM/McMarket_r/results_{c}.jsonl')
        train_data, val_data, test_data = split_dataset(data)

        print(len(train_data), len(val_data), len(test_data))
        print(train_data[0].keys())

        # corpus = single_market(data)
        # bm25 = BM25(corpus)
        # for i in data:
        #     i['top5'] = bm25.get_top_n(i['question'], 5)
        # with open(data_path + f'{c}_questions_bm25.jsonl', 'w', encoding='utf-8') as f:
        #     for line in data:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')




