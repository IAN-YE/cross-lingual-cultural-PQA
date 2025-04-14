from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from collections import defaultdict

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data):
    asin_dict = defaultdict(list)

    for item in data:
        asin = item["asin"]
        asin_dict[asin].append(item)

    return asin_dict

if __name__ == '__main__':
    country2 = ['au','ca','cn','in']
    country1 = ['br','cn','fr','jp','mx']

    # country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'cn']
    country = ['br', 'cn', 'fr', 'jp', 'mx']
    
    data_path = '/home/bcm763/data_PQA/Clothing/'

    auxilary_questions = read_jsonl(data_path + 'us_questions.jsonl')
    auxilary_questions = preprocess_data(auxilary_questions)
    selected_questions = read_jsonl(data_path + 'cn_questions_bm25_similarity.jsonl')

    results = []
    for i in selected_questions:
        q = i['q']
        asin = i['asin']

        questions = auxilary_questions[asin]
        for question in questions:
            if question['question'] == q:
                item_copy = i.copy()
                item_copy['topAnswer_us'] = question['topAnswer']
                results.append(item_copy)
    
    print('Number of results:', len(results))
    # remove duplicates
    import json

    unique_data = list({json.dumps(item, sort_keys=True) for item in results}) 
    results = [json.loads(item) for item in unique_data] 

    print('Number of results after removing duplicates:', len(results))
        
    # Save the results to a new JSONL file
    with open(data_path + 'cn_questions_bm25_similarity_with_answers.jsonl', 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')