from rank_bm25 import BM25Okapi
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from review_AQ.bm25 import main_AQ_bm25 
from review_AQ.CLIR import main
from review_AQ.bert import main_AQ_bertrerank

if __name__ == '__main__':
    # main_AQ_bm25()
    main_AQ_bertrerank()
    # main()