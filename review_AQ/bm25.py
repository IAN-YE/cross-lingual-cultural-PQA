from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict

def preprocess_data(data):
    asin_dict = defaultdict(list)
    seen = defaultdict(set)

    for item in data:
        asin = item["asin"]
        
        review = item["reviewText"]
        if review not in seen[asin]:
            seen[asin].add(review)
            asin_dict[asin].append(item)

    return asin_dict

def single_market(asin_dict, asin):
    return [item["reviewText"] for item in asin_dict.get(asin, [])]

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



# if __name__ == '__main__':
#     country2 = ['au','ca','uk','in']
#     country1 = ['br','cn','fr','jp','mx']

#     country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    
#     data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/'

#     auxilary_review = read_jsonl(data_path + 'McMarket/us_reviews.jsonl')
#     # auxilary_review = preprocess_data(auxilary_review)

#     questions = "May I ask the maximum height of the mount when it's fully extended?"
#     asin = "B01AI2YGK4"
    
#     bm25 = BM25(auxilary_review, asin)

#     top5 = bm25.get_top_n(questions, 5)
#     print(top5)

#     for c in ['cn']:
#         print(c)
#         reivew = read_jsonl(data_path + f'McMarket/{c}_reviews_translated.jsonl')
#         single_review = preprocess_data(reivew)
#         merged_review = preprocess_data(reivew + auxilary_review)
#         questions = read_jsonl(data_path + f'McMarket/{c}_questions_translated.jsonl')
#         questions = [i for i in questions if i['topAnswer']!='']
#         result_examples = read_jsonl(data_path + f'McMarket_LLM/McMarket_r/results_{c}.jsonl')
#         train_data, val_data, test_data = split_dataset(questions)

#         print(len(train_data), len(val_data), len(test_data), len(result_examples))
#         # print(train_data[0].keys())
#         # print(result_examples[0].keys())

#         for i in tqdm(test_data):
#             asin = i['asin']
#             single_corpus = single_market(single_review, asin)
#             merged_corpus = single_market(merged_review, asin)
#             bm25_single = BM25(single_review, asin)
#             top5_single = bm25_single.get_top_n(i['translatedQuestion'], 5)

#             bm25_merged = BM25(merged_review, asin)
#             top5_merged = bm25_merged.get_top_n(i['translatedQuestion'], 5)

#             i['bm25_single_top5'] = top5_single
#             i['bm25_merged_top5'] = top5_merged

#         with open(data_path + f'/AR_bm25/{c}_questions_translated.jsonl', 'w', encoding='utf-8') as f:
#             for line in test_data:
#                 f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
#         hypothesis_single = [i['bm25_single_top5'][0] for i in test_data]
#         reference = [i['topAnswer'] for i in test_data]
#         rougle_result = rouge_score(hypothesis_single, reference)
#         bleu_result = bleu_score(hypothesis_single, reference)
#         bert_result = bert_score(hypothesis_single, reference)
#         print(f"{c} single ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}") 

#         hypothesis_merged = [i['bm25_merged_top5'][0] for i in test_data]
#         rougle_result = rouge_score(hypothesis_merged, reference)
#         bleu_result = bleu_score(hypothesis_merged, reference)
#         bert_result = bert_score(hypothesis_merged, reference)
#         print(f"{c} merged ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")


#         # corpus = single_market(reivew)
#         # bm25 = BM25(corpus)
#         # for i in data:
#         #     i['top5'] = bm25.get_top_n(i['question'], 5)
#         # with open(data_path + f'{c}_questions_bm25.jsonl', 'w', encoding='utf-8') as f:
#         #     for line in data:
#         #         f.write(json.dumps(line, ensure_ascii=False) + '\n')




