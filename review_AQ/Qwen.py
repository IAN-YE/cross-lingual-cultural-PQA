from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Union, Optional
import json

from torch.nn import Module
from tqdm import tqdm

model_name = "../../../../data/public/model/Qwen2.5-7B-Instruct"
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

results = {}

def change_tasting_notes(tasting_notes, type):
    if type == 'English':
        prompt = f"""将这个红酒评论翻译成中文，并且是适合中国人理解的:{tasting_notes}."""
        messages = [{"role": "system", "content": "你是一个红酒评论双语翻译助手."}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            )
    if type == 'Chinese':
        prompt = f"""Translate the wine reviews in English, adapted to an English-speaking consumer:{tasting_notes}."""
        messages = [{"role": "system", "content": "You are a Bilingual Translation Assistant for Red Wine Reviews."}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

import json

with open('combine.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for case in tqdm(data):
    Chinese_reviews = data[case]['Chinese reviews']
    Western_reviews = data[case]['Western_reviews']

    for review in Chinese_reviews:
        Chinese_review = review['review']
        English_review = review['English review']
        if English_review == None:
            new_Chinese_review = None
        else:
            new_Chinese_review = change_tasting_notes(English_review, 'English')
        new_English_review = change_tasting_notes(Chinese_review, 'Chinese')

    # Chinese_translation = []
    cnt_west = 0
    for review in Western_reviews:
        if cnt_west == 5:
            break
        Western_review = review['review']
        chinese_translation = change_tasting_notes(Western_review, 'English')
        # Chinese_translation.append({'review': Western_review, 'translation': chinese_translation})
        review['translation'] = chinese_translation
        cnt_west += 1

    results[case] = {
        'name': case,
        'Chinese pairs': {
            'original': Chinese_review,
            'new': new_English_review
        },
        'English pairs': {
            'original': English_review,
            'new': new_Chinese_review
        },
        'Western reviews': Western_reviews
    }
