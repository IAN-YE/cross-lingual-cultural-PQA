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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

results = {}


