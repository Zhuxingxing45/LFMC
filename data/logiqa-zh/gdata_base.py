import json
from tqdm import tqdm
import ast

with open('zh_train.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    context = example['context']
    query = example['query']
    options = example['options']
    correct_option = example['correct_option']
    if correct_option == 0:
        answer = 'A'
    elif correct_option == 1:
        answer = 'B'
    elif correct_option == 2:
        answer = 'C'
    else:
        answer = 'D'
    
    prompt = "给定以下背景信息：\n" + context + f"\n对于以下问题：{query}\n  A){options[0]}  B){options[1]} C){options[2]} D){options[3]}\n" + "请提供正确的选项。"
    # reasoning_process = example['reasoning_process']

    data = {
                "conversation": [
                    {
                        "system":"你是一名逻辑学家。请根据给定的背景信息和问题从选项中选择正确的答案。",
                        "input": prompt,
                        "output":f"正确的答案是:{answer}) {options[correct_option]}"
                    }
                ]
            }
    dataset.append(data)

with open('logiqa-zh_fintuing_data_formatted_base.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
