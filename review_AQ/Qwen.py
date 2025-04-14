from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import os
import json

from torch.nn import Module
from tqdm import tqdm

def get_prompt(question: str) -> str:
    prompt = f"""
                Question: "{question}"

                Would people from different regions or cultures likely give different answers to this question? (Yes/No)

                Explanation: (Briefly explain your reasoning)
            """
    return prompt

def generate_response(prompt: str) -> str:
    inputs = tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a Translation Assistant."},
                 {"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True)

    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model_name = "/home/bcm763/Models/Qwen2.5-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

results = {}

if __name__ == '__main__':
    print("Model loaded")
