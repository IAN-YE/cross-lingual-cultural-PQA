from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils import read_jsonl, preprocess_data, split_dataset
from utils import bleu_score, rouge_score, bert_score


def single_market(asin_dict, asin, type='translate'):
    if type == 'translate':
        return [
            item.get("translatedReview", item.get("reviewText", ""))
            for item in asin_dict.get(asin, [])
        ]
    else:
        return [item.get("reviewText", "") for item in asin_dict.get(asin, [])]


class BM25:
    def __init__(self, asin_dict, asin):
        self.corpus = single_market(asin_dict, asin, type='translate')
        self.tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_top_n(self, query, n=5):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        # print(doc_scores)
        top_n = [self.corpus[i] for i in np.argsort(doc_scores)[::-1][:n]]
        return top_n

def main_AQ_bm25():
    print("BM25")
    print("Clothing")
    country2 = ['au','ca','uk','in']
    country1 = ['cn','jp']

    country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    
    data_path = '/home/bcm763/data_PQA/Clothing/'

    auxilary_review = read_jsonl(data_path + 'us_reviews.jsonl')
    # auxilary_review = preprocess_data(auxilary_review)

    # questions = "May I ask the maximum height of the mount when it's fully extended?"
    # asin = "B01AI2YGK4"
    
    # bm25 = BM25(auxilary_review, asin)

    # top5 = bm25.get_top_n(questions, 5)
    # print(top5)

    for c in country1:
        print(c)
        print("Clothing")
        reivew = read_jsonl(data_path + f'{c}_reviews_translated.jsonl')
        single_review = preprocess_data(reivew, type='translate')
        merged_review = preprocess_data(reivew + auxilary_review, type='translate')
        questions_original = read_jsonl(data_path + f'{c}_questions_translated.jsonl')
        questions_new = [i for i in questions_original if i['topAnswer']!='']
        # result_examples = read_jsonl(data_path + f'McMarket_LLM/McMarket_r/results_{c}.jsonl')
        train_data, val_data, test_data = split_dataset(questions_new)

        print(len(train_data), len(val_data), len(test_data))
        print(len(questions_original), len(questions_new))
        # print(train_data[0].keys())
        # print(result_examples[0].keys())

        for i in questions_new:
            asin = i['asin']
            # single_corpus = single_market(single_review, asin)
            # merged_corpus = single_market(merged_review, asin)
            bm25_single = BM25(single_review, asin)
            top5_single = bm25_single.get_top_n(i['translatedQuestion'], 5)

            bm25_merged = BM25(merged_review, asin)
            top5_merged = bm25_merged.get_top_n(i['translatedQuestion'], 5)

            i['bm25_single_top5'] = top5_single
            i['bm25_merged_top5'] = top5_merged

        # with open(data_path + f'/AR_bm25/{c}_questions_translated.jsonl', 'w', encoding='utf-8') as f:
        #     for line in test_data:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        hypothesis_single = [i['bm25_single_top5'][0] for i in test_data]
        reference = [i['translatedAnswer'] for i in test_data]
        rougle_result = rouge_score(hypothesis_single, reference)
        bleu_result = bleu_score(hypothesis_single, reference)
        bert_result = bert_score(hypothesis_single, reference)
        print(f"{c} single ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}") 

        hypothesis_merged = [i['bm25_merged_top5'][0] for i in test_data]
        rougle_result = rouge_score(hypothesis_merged, reference)
        bleu_result = bleu_score(hypothesis_merged, reference)
        bert_result = bert_score(hypothesis_merged, reference)
        print(f"{c} merged ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")




