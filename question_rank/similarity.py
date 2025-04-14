from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from bm25 import preprocess_data, BM25, read_jsonl, single_market
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import json

model = SentenceTransformer('sentence-transformers/LaBSE')

if __name__ == '__main__':
    country2 = ['au','ca','uk','in']
    country1 = ['br','cn','fr','jp','mx']

    # country = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    country = ['br', 'cn', 'fr', 'jp', 'mx']
    
    data_path = '/home/bcm763/data_PQA/Clothing/'

    auxilary_questions = read_jsonl(data_path + 'cn_questions_bm25.jsonl')
    

    for c in ['cn']:
        print(c)
        data = read_jsonl(data_path + f'{c}_questions_bm25.jsonl')
        
        results = []

        sim = []

        similarities = []

        for i in data:
            question = i['question']
            translated_question = i['translatedQuestion']
            for q,s in zip(i['top50'], i['top50_score']):
                embeddings = model.encode([question, q])
                similarity = util.cos_sim(embeddings[0], embeddings[1])

                embeddings_translated = model.encode([translated_question, q])
                similarity_translated = util.cos_sim(embeddings_translated[0], embeddings_translated[1])

                similarities.append([similarity.item(), similarity_translated.item()])

                # sim.append(similarity.item())
                # if similarity.item() > 0.8:
                #     item_copy = copy.deepcopy(i)
                #     item_copy['q'] = q
                #     item_copy['s'] = s
                #     item_copy.pop('top50', None)
                #     item_copy.pop('top50_score', None)
                #     results.append(item_copy)
        # print s stats
        import numpy as np
        similarities = np.array(similarities)
        print('Similarity stats:')
        print('Mean:', np.mean(similarities[:, 0]))
        print('Std:', np.std(similarities[:, 0]))
        print('Max:', np.max(similarities[:, 0]))
        print('Min:', np.min(similarities[:, 0]))
        print('Mean translated:', np.mean(similarities[:, 1]))
        print('Std translated:', np.std(similarities[:, 1]))
        print('Max translated:', np.max(similarities[:, 1]))
        print('Min translated:', np.min(similarities[:, 1]))

        from matplotlib import pyplot as plt
        score_A = [pair[0] for pair in similarities]
        score_B = [pair[1] for pair in similarities]

        plt.scatter(score_A, score_B)
        plt.plot([0, 1], [0, 1], 'r--', label='A = B')
        plt.xlabel('Original Similarity')
        plt.ylabel('Translated Similarity')
        plt.title('Original vs Translated Similarities')
        plt.legend()
        plt.grid(True)
        plt.savefig('similarity_scatter_plot.png')
        plt.show()



        # with open(data_path + f'{c}_questions_bm25_similarity.jsonl', 'w', encoding='utf-8') as f:
        #     for line in results:
        #         f.write(json.dumps(line, ensure_ascii=False) + '\n')



