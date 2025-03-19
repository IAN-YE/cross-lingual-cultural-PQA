from rank_bm25 import BM25Okapi
import numpy as np
import json
from sentence_transformers import CrossEncoder
from scipy.special import softmax
import numpy as np
import os
from tqdm import tqdm
import math
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from bm25 import preprocess_data, BM25, read_jsonl, single_market
import torch
import torch.nn.functional as F
import torch.nn as nn

model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)
class Model:
    def __init__(self, k):
        self.k = k
        self.item_size = 50

    def __call__(self, users):
        res = np.random.randint(0, self.item_size, users.shape[0] * self.k)
        return res.reshape((users.shape[0], -1))


class BM25BERTReRanker:
    def __init__(self, bm25, model_name="amberoad/bert-multilingual-passage-reranking-msmarco", k=5):
        self.bm25 = bm25
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.k = k
    
    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def rerank_with_bert(self, query, bm25_results):
        query_embedding = self.encode_text(query)
        doc_embeddings = torch.cat([self.encode_text(doc[0]) for doc in bm25_results]) 
        similarities = nn.CosineSimilarity(query_embedding, doc_embeddings).squeeze(0)

        sorted_indices = similarities.argsort(descending=True)
        reranked_results = [(bm25_results[i][0], similarities[i].item()) for i in sorted_indices]
        return reranked_results
    
    def get_top_n(self, query, n=5):
        bm25_results = self.bm25.get_top_n(query, n=self.k)
        reranked_results = self.rerank_with_bert(query, bm25_results)
        return reranked_results[:n], bm25_results
    
    if __name__ == '__main__':
        data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/'

        auxilary_data = read_jsonl(data_path + 'McMarket/us_questions.jsonl')
        auxilary_data = preprocess_data(auxilary_data)
        country = ['cn']

        questions = "Does this tablet have 32g of memory?"
        asin = "B0171BS9CG"
        corpus = single_market(auxilary_data, asin)
        print(corpus[0])
        
        bm25 = BM25(corpus, asin)

        top5 = bm25.get_top_n(questions, 5)
        print(top5)

        reranker = BM25BERTReRanker(bm25)
        reranked_results, bm25_results = reranker.get_top_n(questions, 5)
        print(reranked_results)
        print(bm25_results)

