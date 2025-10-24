import json
from tqdm import tqdm
import ast

# with open('Reclor_fintuing_data_v1.json', 'r') as f:
with open('train_4.64k.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    context = example['context']
    question = example['question']
    answers = example['answers']
    label = example['label']
    if label == 0:
        answer = 'A'
    elif label == 1:
        answer = 'B'
    elif label == 2:
        answer = 'C'
    else:
        answer = 'D'
    
    prompt = "Given the following context:\n" + context + f"\nFor the following question:{question}\n Which of the following options is correct? A){answers[0]}  B){answers[1]}  C){answers[2]}  D){answers[3]}\n" + "Please provide the correct option."
    # reasoning_process = example['reasoning_process']

    data = {
                "conversation": [
                    {
                        "system":"You are a logician. Please select the correct answer from the options based on the given context and question.",
                        "input": prompt,
                        "output":f"The answer is:{answer}) {answers[label]}"
                    }
                ]
            }
    dataset.append(data)

with open('Reclor_fintuing_data_formatted_base.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
