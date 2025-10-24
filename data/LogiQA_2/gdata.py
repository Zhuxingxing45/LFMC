import json
from tqdm import tqdm
import ast

with open('LogiQA_fintuing_data_v1.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    premise = example['premise']
    hypothesis = example['hypothesis']
    label = example['label']
    if label == 'entailment':
        answer = 'A'
    elif label == 'not-entailment':
        answer = 'B'

    
    prompt = "Given the following premises:\n" + premise + f"\nFor the following hypothesis:{hypothesis} \n Which of the following options is correct? A)entailment, B)not-entailment \n" + "Please provide the correct option and the reasoning process to verify this conclusion."
    reasoning_process = example['reasoning_process']

    data = {
                "conversation": [
                    {
                        "system":"You are a logician who can identify logical fallacies in sentences, Please select the correct answer from the options based on the given context and question, and provide the reasoning process.",
                        "input": prompt,
                        "output":reasoning_process +f"\nTherefore, the answer is:{answer}) {label}.",
                        "answer":answer
                    }
                ]
            }
    dataset.append(data)

with open('LogiQA_fintuing_data_formatted_v1.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
