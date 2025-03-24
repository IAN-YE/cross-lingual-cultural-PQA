from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict

def preprocess_data(data):
    asin_dict = defaultdict(list)
    seen_questions = defaultdict(set)

    for item in data:
        asin = item["asin"]
        question = item["question"]

        if question not in seen_questions[asin]:
            seen_questions[asin].add(question)
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



if __name__ == '__main__':
    country2 = ['au','ca','uk','in']
    country1 = ['br','cn','fr','jp','mx']

    country = ['cn']
    
    data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/'

    auxilary_data = read_jsonl(data_path + 'McMarket/us_questions.jsonl')
    auxilary_data = preprocess_data(auxilary_data)

    questions = "Does this tablet have 32g of memory?"
    asin = "B0171BS9CG"
    
    bm25 = BM25(auxilary_data, asin)

    top5 = bm25.get_top_n(questions, 5)
    print(top5)

    # for c in country:
    #     print(c)
    #     data = read_jsonl(data_path + f'McMarket/{c}_questions_translated.jsonl')
    #     result_examples = read_jsonl(data_path + f'McMarket_LLM/McMarket_r/results_{c}.jsonl')
    #     train_data, val_data, test_data = split_dataset(data)

    #     print(len(train_data), len(val_data), len(test_data), len(result_examples))
    #     # print(train_data[0].keys())
    #     # print(result_examples[0].keys())

    #     data_questions = {d["translatedQuestion"] for d in result_examples}
    #     print(len(data_questions))

    #     results = []

    #     for i in tqdm(data_questions):
    #         top5 = bm25.get_top_n(i, 5)
    #         results.append({"question": i, "top5": top5})

    #     with open(data_path + f'{c}_questions_bm25.jsonl', 'w', encoding='utf-8') as f:
    #         for line in results:
    #             f.write(json.dumps(line, ensure_ascii=False) + '\n')
        

        # corpus = single_market(data)
        # bm25 = BM25(corpus)
        # for i in data:
        #     i['top5'] = bm25.get_top_n(i['question'], 5)
        # with open(data_path + f'{c}_questions_bm25.jsonl', 'w', encoding='utf-8') as f:
        #     for line in data:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')




