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
    print("Clothing")
    print("CLIR-jinaai/jina-reranker-v2-base-multilingual")
    country2 = ['cn', 'jp']
    country1 = ['cn','de', 'in', 'jp']

    country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    
    data_path = '/home/bcm763/data_PQA/Clothing/'

    auxilary_review = read_jsonl(data_path + 'us_reviews.jsonl')

    CLIR_model = CLIR()

    for c in country2:
        print(c)
        reivew = read_jsonl(data_path + f'{c}_reviews_translated.jsonl')
        single_review = preprocess_data(reivew, type='translate')
        merged_review = preprocess_data(reivew + auxilary_review, type='translate')
        questions_original = read_jsonl(data_path + f'{c}_questions_translated.jsonl')
        questions_new = [i for i in questions_original if i['topAnswer']!='']
        # result_examples = read_jsonl(data_path + f'McMarket_LLM/McMarket_r/results_{c}.jsonl')
        train_data, val_data, test_data = split_dataset(questions_new)

        print(f" train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
        print(f"questions_original: {len(questions_original)}, questions_new: {len(questions_new)}")
        print(f"single_review: {len(single_review)}, merged_review: {len(merged_review)}")
        for i in test_data:
            asin = i['asin']
            single_corpus = single_market(single_review, asin, type='no-translate')
            merged_corpus = single_market(merged_review, asin, type='no-translate')

            single_pairs = [[i['question'], doc] for doc in single_corpus]
            merged_pairs = [[i['question'], doc] for doc in merged_corpus]

            # both single and merged are not empty
            if not single_pairs or not merged_pairs:
                continue
            single_scores = CLIR_model.compute_score(single_pairs)
            merged_scores = CLIR_model.compute_score(merged_pairs)

            single_results = [single_pairs[i][1] for i in single_scores[:5]]
            merged_results = [merged_pairs[i][1] for i in merged_scores[:5]]

            i['bm25_single_top5'] = single_results
            i['bm25_merged_top5'] = merged_results
        
        # with open(data_path + f'/AR_bm25/{c}_CLIR.jsonl', 'w', encoding='utf-8') as f:
        #     for line in questions_new:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        hypothesis_single = []
        hypothesis_merged = []
        reference = []

        for i in questions_new:
            if 'bm25_single_top5' in i and i['bm25_single_top5']:
                hypothesis_single.append(i['bm25_single_top5'][0])
                hypothesis_merged.append(i['bm25_merged_top5'][0])
                reference.append(i['translatedAnswer']) 

        rougle_result = rouge_score(hypothesis_single, reference)
        bleu_result = bleu_score(hypothesis_single, reference)
        bert_result = bert_score(hypothesis_single, reference)
        print(f"{c} single ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}") 


        rougle_result = rouge_score(hypothesis_merged, reference)
        bleu_result = bleu_score(hypothesis_merged, reference)
        bert_result = bert_score(hypothesis_merged, reference)
        print(f"{c} merged ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")