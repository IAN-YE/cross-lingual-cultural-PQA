from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from bm25 import preprocess_data, BM25, read_jsonl, single_market
import torch
import torch.nn.functional as F
import torch.nn as nn

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


class BM25BERTReRanker:
    def __init__(self, bm25, model_name="amberoad/bert-multilingual-passage-reranking-msmarco", k=5):
        self.bm25 = bm25
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)
        
        self.k = k
    
    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def rerank_with_bert(self, query, bm25_results):
        if not bm25_results:
            return []
        
        query_embedding = self.encode_text(query) # (1, D)
        doc_embeddings = torch.cat([self.encode_text(doc[0]) for doc in bm25_results]) # (N, D)
        print(query_embedding.shape, doc_embeddings.shape)
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(query_embedding, doc_embeddings)

        sorted_indices = similarities.argsort(descending=True)
        print(similarities)
        print(sorted_indices)
        reranked_results = [(bm25_results[i], similarities[i].item()) for i in sorted_indices]
        return reranked_results
    
    def rerank_with_bert_scores(self, query, bm25_results):
        inputs = self.tokenizer(bm25_results, query, return_tensors="pt", padding=True, truncation=True)
        print(inputs.shape)
        with torch.no_grad():
            outputs = self.score_model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)
        sorted_indices = scores.argsort(descending=True)
        reranked_results = [(bm25_results[i], scores[i].item()) for i in sorted_indices]
        return reranked_results

    def rerank(self, query, topAnswer, bm25_results):
        if not bm25_results:
            return []
        pairs = []
        for k in bm25_results:
            pairs.append((query+' '+topAnswer,k))
        scores = self.model.predict(pairs)
        scores = softmax(scores,axis=1)
        scores = np.argmax(scores,axis=0)
        hypo_0 = bm25_results[scores[0]]
        hypo_1 = bm25_results[scores[1]]
        
        return hypo_0, hypo_1
    
    def get_top_n(self, query, topAnswer, n=5):
        bm25_results = self.bm25.get_top_n(query, n=n)
        rerank_res_1, rerank_res_2 = self.rerank(query, topAnswer, bm25_results)

        return rerank_res_1, rerank_res_2
    
def main_AQ_bertrerank():
    print("Bert ReRank")
    print("McMarket")
    country2 = ['cn', 'jp']
    country1 = ['de', 'in']

    country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    
    data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/McMarket/'

    auxilary_review = read_jsonl(data_path + 'us_reviews.jsonl')
    # auxilary_review = preprocess_data(auxilary_review)

    # questions = "May I ask the maximum height of the mount when it's fully extended?"
    # asin = "B01AI2YGK4"
    
    # bm25 = BM25(auxilary_review, asin)

    # top5 = bm25.get_top_n(questions, 5)
    # print(top5)

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
        # print(train_data[0].keys())
        # print(result_examples[0].keys())

        hypothesis_single_0 = []
        hypothesis_single_1 = []
        hypothesis_merged_0 = []
        hypothesis_merged_1 = []

        reference = []

        for i in questions_new:
            asin = i['asin']
            answer = i['translatedAnswer']
            if len(single_market(single_review, asin, type='translate')) == 0:
                continue
            
            # single_corpus = single_market(single_review, asin)
            # merged_corpus = single_market(merged_review, asin)

            bm25_single = BM25(single_review, asin)
            rerank_single = BM25BERTReRanker(bm25_single)
            top5_single_0, top5_single_1  = rerank_single.get_top_n(i['translatedQuestion'], answer, 5)

            bm25_merged = BM25(merged_review, asin)
            rerank_merged = BM25BERTReRanker(bm25_merged)
            top5_merged_0, top5_merged_1 = rerank_merged.get_top_n(i['translatedQuestion'], answer, 5)

            hypothesis_single_0.append(top5_single_0)
            hypothesis_single_1.append(top5_single_1)
            hypothesis_merged_0.append(top5_merged_0)
            hypothesis_merged_1.append(top5_merged_1)
            reference.append(i['translatedAnswer'])

        
        rougle_result = rouge_score(hypothesis_single_0, reference)
        bleu_result = bleu_score(hypothesis_single_0, reference)
        bert_result = bert_score(hypothesis_single_0, reference)
        print(f"{c} single_0 ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")

        rougle_result = rouge_score(hypothesis_single_1, reference)
        bleu_result = bleu_score(hypothesis_single_1, reference)
        bert_result = bert_score(hypothesis_single_1, reference)
        print(f"{c} single_1 ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")

        rougle_result = rouge_score(hypothesis_merged_0, reference)
        bleu_result = bleu_score(hypothesis_merged_0, reference)
        bert_result = bert_score(hypothesis_merged_0, reference)
        print(f"{c} merged_0 ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")

        rougle_result = rouge_score(hypothesis_merged_1, reference)
        bleu_result = bleu_score(hypothesis_merged_1, reference)
        bert_result = bert_score(hypothesis_merged_1, reference)
        print(f"{c} merged_1 ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")




