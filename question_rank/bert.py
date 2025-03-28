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

# model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

class BM25BERTReRanker:
    def __init__(self, bm25, model_name="amberoad/bert-multilingual-passage-reranking-msmarco", k=5):
        self.bm25 = bm25
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
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

    
    
    def get_top_n(self, query, n=5):
        bm25_results = self.bm25.get_top_n(query, n=self.k)
        reranked_results = self.rerank_with_bert(query, bm25_results)
        reranked_score_results = self.rerank_with_bert_scores(query, bm25_results)
        return reranked_results[:n], reranked_score_results[:n] ,bm25_results
    
# if __name__ == '__main__':
#     data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/'

#     auxilary_data = read_jsonl(data_path + 'McMarket/us_questions.jsonl')
#     auxilary_data = preprocess_data(auxilary_data)
#     country = ['cn']

#     questions = "Does this tablet have 32g of memory?"
#     asin = "B0171BS9CG"
    
#     bm25 = BM25(auxilary_data, asin)

#     top5 = bm25.get_top_n(questions, 5)
#     print(top5)

#     reranker = BM25BERTReRanker(bm25)
#     reranked_results, reranked_scores_results, bm25_results = reranker.get_top_n(questions, 5)
#     print(reranked_results)
#     print(reranked_scores_results)
#     print(bm25_results)