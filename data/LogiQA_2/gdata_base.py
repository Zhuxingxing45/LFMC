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

    
    prompt = "Given the following premises:\n" + premise + f"\nFor the following hypothesis:{hypothesis} \n Which of the following options is correct? A)entailment, B)not-entailment \n" + "Please provide the correct option."
    #reasoning_process = example['reasoning_process']

    data = {
                "conversation": [
                    {
                        "system":"You are a logician. Please select the correct answer from the options based on the given context and question.",
                        "input": prompt,
                        "output":f"The answer is:{answer}) {label}."
                    }
                ]
            }
    dataset.append(data)

with open('LogiQA_fintuing_data_formatted_base.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
