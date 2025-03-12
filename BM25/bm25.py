from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split

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

if __name__ == '__main__':
    corpus = [
        "Hello there",
        "How are you doing?",
        "The weather is great today",
        "An apple a day keeps the doctor away",
        "The quick brown fox jumps over the lazy dog"
    ]

    bm25 = BM25(corpus)
    query = "the"
    top_n = bm25.get_top_n(query, n=2)
    print(top_n)
