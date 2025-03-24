import evaluate
import json

def bleu_score(hypothesis, reference):
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
    return bleu_score['bleu']

def rouge_score(hypothesis, reference):
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=hypothesis, references=reference)
    return rouge_score['rouge2'].mid.fmeasure

def bert_score(hypothesis, reference):
    bert = evaluate.load("bertscore", module_type="metric")
    results = bert.compute(predictions=hypothesis, references=reference, model_type="distilbert-base-uncased", lang="en")
    return sum(results['f1'])/len(results['f1'])

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':
    countries = ['au', 'br', 'ca', 'cn', 'fr', 'in', 'jp', 'mx', 'uk']
    data_path = '/home/bcm763/data_PQA/McMarket/McMarket_all/McMarket_LLM/'
    for c in countries:
        contents = read_jsonl(data_path + f'McMarket_r/results_{c}.jsonl')
        contents = [i for i in contents if i['topAnswer']!='']
        hypothesis = [i['bm25_top5'][0] for i in contents]
        reference = [i['topAnswer'] for i in contents]
        rougle_result = rouge_score(hypothesis, reference)
        bleu_result = bleu_score(hypothesis, reference)
        bert_result = bert_score(hypothesis, reference)
        print(f"{c} ROUGE: {rougle_result} BLEU: {bleu_result} BERT: {bert_result}")


    