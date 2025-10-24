import json
from tqdm import tqdm
import ast

with open('FOLIO_fintuing_data_v1.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    premises = example['premises']
    conclusion = example['conclusion']
    label = example['label']
    if label == 'True':
        answer = 'A'
    elif label == 'False':
        answer = 'B'
    else:
        answer = 'C'
    
    prompt = "Given the following premises:\n" + premises + f"\nFor the following hypothesis:{conclusion}\nWhich of the following options is correct? A)True, B)False, C)Uncertain \n" + "Please provide the correct option."
    reasoning_process = example['reasoning_process']

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

with open('FOLIO_fintuing_data_formatted_base.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
