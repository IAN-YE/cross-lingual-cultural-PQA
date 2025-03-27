from transformers import AutoModelForSequenceClassification
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils import read_jsonl, preprocess_data, split_dataset
from utils import bleu_score, rouge_score, bert_score
from bm25 import BM25, single_market
import json

class CLIR:
    def __init__(self, model="jinaai/jina-reranker-v2-base-multilingual"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def compute_score(self, sentence_pairs, max_length=512):
        scores = self.model.compute_score(sentence_pairs, max_length=max_length)
        sorted_indices = np.argsort(scores)[::-1]

        return sorted_indices


def main():
    print("CLIR-jinaai/jina-reranker-v2-base-multilingual")
    country2 = ['au','ca','uk','in']
    country1 = ['br','cn','fr','jp','mx']

    country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    
    data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/'

    auxilary_review = read_jsonl(data_path + 'McMarket/us_reviews.jsonl')

    CLIR_model = CLIR()

    for c in country1:
        print(c)
        reivew = read_jsonl(data_path + f'McMarket/{c}_reviews_translated.jsonl')
        single_review = preprocess_data(reivew, type='no-translate')
        merged_review = preprocess_data(reivew + auxilary_review, type='no-translate')
        questions_original = read_jsonl(data_path + f'McMarket/{c}_questions_translated.jsonl')
        questions_new = [i for i in questions_original if i['topAnswer']!='']
        result_examples = read_jsonl(data_path + f'McMarket_LLM/McMarket_r/results_{c}.jsonl')
        train_data, val_data, test_data = split_dataset(questions_new)

        print(len(train_data), len(val_data), len(test_data), len(result_examples))
        print(len(questions_original), len(questions_new))
        # print(train_data[0].keys())
        # print(result_examples[0].keys())

        for i in test_data:
            asin = i['asin']
            single_corpus = single_market(single_review, asin, type='no-translate')
            merged_corpus = single_market(merged_review, asin, type='no-translate')

            single_pairs = [[i['question'], doc] for doc in single_corpus]
            merged_pairs = [[i['question'], doc] for doc in merged_corpus]

            single_scores = CLIR_model.compute_score(single_pairs)
            merged_scores = CLIR_model.compute_score(merged_pairs)

            single_results = [single_pairs[i][1] for i in single_scores[:5]]
            merged_results = [merged_pairs[i][1] for i in merged_scores[:5]]

            i['bm25_single_top5'] = single_results
            i['bm25_merged_top5'] = merged_results
        
        # with open(data_path + f'/AR_bm25/{c}_CLIR.jsonl', 'w', encoding='utf-8') as f:
        #     for line in questions_new:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        hypothesis_single = [i['bm25_single_top5'][0] for i in test_data]
        reference = [i['topAnswer'] for i in test_data]
        rougle_result = rouge_score(hypothesis_single, reference)
        bleu_result = bleu_score(hypothesis_single, reference)
        bert_result = bert_score(hypothesis_single, reference)
        print(f"{c} single ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}") 

        hypothesis_merged = [i['bm25_merged_top5'][0] for i in test_data]
        rougle_result = rouge_score(hypothesis_merged, reference)
        bleu_result = bleu_score(hypothesis_merged, reference)
        bert_result = bert_score(hypothesis_merged, reference)
        print(f"{c} merged ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")


        # corpus = single_market(reivew)
        # bm25 = BM25(corpus)
        # for i in data:
        #     i['top5'] = bm25.get_top_n(i['question'], 5)
        # with open(data_path + f'{c}_questions_bm25.jsonl', 'w', encoding='utf-8') as f:
        #     for line in data:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')