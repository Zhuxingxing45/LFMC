import json
from tqdm import tqdm
import ast

with open('/home/23_zxx/project/LLM_correcting/outputs/logic_corrections/LogiQA_v2_gpt-4.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    premise = example['premise']
    hypothesis = example['hypothesis']
    # label = example['label']
    # if label == 'entailment':
    #     answer = 'A'
    # elif label == 'not-entailment':
    #     answer = 'B'
    reference = example['reference']
    generate_answer = example['generate_answer']

    
    prompt = "Given the following premises:\n" + premise + f"\nFor the following hypothesis:{hypothesis}\nWhich of the following options is correct? A)entailment, B)not-entailment\n" + "Please provide the correct option and the reasoning process to verify this conclusion.\n" + f"The original reasoning process is as follows:\n {generate_answer}\n" + f"However, the correct option is{reference}.Please identify and explain the mistakes in the original reasoning process, then correct these mistakes and provide the corrected final answer.Please provide the explicit option in the final line." 
 

    data = {
                "conversation": [
                    {
                        "system":"Given a set of premises, a conclusion, and a reasoning process for the validity of this conclusion, the task is to identify whether the reasoning process is correct. If it is not correct, find the erroneous steps, explain the reasons for the errors, and correct the original solution from the erroneous steps. The response should be as concise as possible.",
                        "input": prompt,
                        "output":example['raw_logic_corrections'][0]
                    }
                ]
            }
    dataset.append(data)

with open('LogiQA_correct_fintuing_data_formatted.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
