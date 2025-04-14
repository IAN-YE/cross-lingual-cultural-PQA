from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import gzip
from langdetect import detect
import random

def preprocess_data(data):
    asin_dict = defaultdict(list)

    for item in data:
        asin = item["asin"]
        asin_dict[asin].append(item)

    return asin_dict

def single_market(asin_dict, asin):
    return [item["question"] for item in asin_dict.get(asin, [])]

class BM25:
    def __init__(self, asin_dict, asin):
        self.corpus = single_market(asin_dict, asin)
        self.tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_top_n(self, query, n=5):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        # print(doc_scores)
        top_n = [self.corpus[i] for i in np.argsort(doc_scores)[::-1][:n]]
        top_n_score = [doc_scores[i] for i in np.argsort(doc_scores)[::-1][:n]]
        return top_n, top_n_score

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

def read_raw_data(file):
    df = pd.read_json(file, lines=True, compression='gzip', encoding='utf-8')
    return df.to_dict(orient='records')

if __name__ == '__main__':
    country2 = ['au','ca','uk','in']
    country1 = ['br','cn','fr','jp','mx']

    country = ['au', 'br', 'ca', 'cn', 'de', 'es', 'fr', 'in_', 'it', 'jp', 'mx']
    
    data_path = '/home/bcm763/data_PQA/XAmazon/'

    # auxilary_questions = read_raw_data(data_path + 'sg_questions.jsonl.gz')
    # print(auxilary_questions)
    # print(len(auxilary_questions))
    # auxilary_questions = preprocess_data(auxilary_questions)
    # print(len(auxilary_questions))

    au = read_raw_data(data_path + 'au_questions.jsonl.gz')
    br = read_raw_data(data_path + 'br_questions.jsonl.gz')
    ca = read_raw_data(data_path + 'ca_questions.jsonl.gz')
    cn = read_raw_data(data_path + 'cn_questions.jsonl.gz')
    de = read_raw_data(data_path + 'de_questions.jsonl.gz')
    es = read_raw_data(data_path + 'es_questions.jsonl.gz')
    fr = read_raw_data(data_path + 'fr_questions.jsonl.gz')
    in_ = read_raw_data(data_path + 'in_questions.jsonl.gz')
    it = read_raw_data(data_path + 'it_questions.jsonl.gz')
    jp = read_raw_data(data_path + 'jp_questions.jsonl.gz')
    mx = read_raw_data(data_path + 'mx_questions.jsonl.gz')
    # nl = read_raw_data(data_path + 'nl_questions.jsonl.gz')
    # sa = read_raw_data(data_path + 'sa_questions.jsonl.gz')
    # sg = read_raw_data(data_path + 'sg_questions.jsonl.gz')
    # tr = read_raw_data(data_path + 'tr_questions.jsonl.gz')
    # uk = read_raw_data(data_path + 'uk_questions.jsonl.gz')
    # us = read_raw_data(data_path + 'us_questions.jsonl.gz')

    print('Number of questions in each country:')
    print('au:', len(au))
    print('br:', len(br))
    print('ca:', len(ca))
    print('cn:', len(cn))
    print('de:', len(de))
    print('es:', len(es))
    print('fr:', len(fr))
    print('in:', len(in_))
    print('it:', len(it))
    print('jp:', len(jp))
    print('mx:', len(mx))
    # print('nl:', len(nl))
    # print('sa:', len(sa))
    # print('sg:', len(sg))
    # print('tr:', len(tr))
    # print('uk:', len(uk))
    # print('us:', len(us))

    data = {c: globals()[c] for c in country}

    for c in country:
        print(c)
        country_data = data[c]

        random_select_100 = random.sample(country_data, 100)

        langs = {}

        for i in random_select_100:
            asin = i['asin']
            question = i['question']
            lang = detect(question)
            if lang not in langs:
                langs[lang] = 1
            else:
                langs[lang] += 1
        print(langs)


        # auxilary_data = pd.concat([data[cc] for cc in country if cc != c], ignore_index=True)
        # auxilary_questions = preprocess_data(auxilary_data)

        # results = []

        # for i in data:
        #     asin = i['asin']
        #     question = i['question']
        #     bm25 = BM25(auxilary_questions, asin)
        #     top50, top50_score = bm25.get_top_n(question, 50)
        #     i['top50'] = top50
        #     i['top50_score'] = top50_score
        #     results.append(i)

        # with open(data_path + f'BM25/{c}_questions_bm25.jsonl', 'w', encoding='utf-8') as f:
        #     for line in results:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')