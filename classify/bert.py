import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
import json

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                print(f"Error decoding JSON: {line}")
                continue
    return data    

data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/McMarket/'
country1 = ['br', 'cn', 'fr', 'jp', 'mx']
country2 = ['au', 'ca', 'de', 'es', 'in', 'it', 'nl', 'sa']

review_data = []

# read data
for c in country1:
    data = read_jsonl(data_path + f'{c}_reviews_translated.jsonl')
    review_data.extend([[i['translatedReview'], c] for i in data])

for c in country2:
    data = read_jsonl(data_path + f'{c}_reviews.jsonl')
    review_data.extend([[i['reviewText'], c] for i in data])

# convert to DataFrame
df = pd.DataFrame(review_data, columns=['text', 'label'])

print(df['label'].value_counts())

# random select 1000 samples from each class
df = df.groupby('label').apply(lambda x: x.sample(min(1000, len(x)))).reset_index(drop=True)
print(df['label'].value_counts())

# map labels to integers
label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
print(label_map)
df['label'] = df['label'].map(label_map)

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 3. 划分训练/验证集
split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

# 4. 加载模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 6. 评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# 7. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

