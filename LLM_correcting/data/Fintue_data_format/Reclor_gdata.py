import json
from tqdm import tqdm
import ast

with open('/home/23_zxx/project/LLM_correcting/outputs/logic_corrections/Reclor_gpt-4.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    context = example['context']
    question = example['question']
    answers = example['answers']
    # label = example['label']
    # if label == 0:
    #     answer = 'A'
    # elif label == 1:
    #     answer = 'B'
    # elif label == 2:
    #     answer = 'C'
    # else:
    #     answer = 'D'
    reference = example['reference']
    generate_answer = example['generate_answer']
    
    prompt = "Given the following context:\n" + context + f"\nFor the following question:{question}\n Which of the following options is correct? A){answers[0]}, B){answers[1]}, C){answers[2]}, D){answers[3]}\n" + "Please provide the correct option and the reasoning process to verify this conclusion.\n" + f"The original reasoning process is as follows:\n {generate_answer}\n" + f"However, the correct option is{reference}.Please identify and explain the mistakes in the original reasoning process, then correct these mistakes and provide the corrected final answer.Please provide the explicit option in the final line."

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

with open('Reclor_correct_fintuing_data_formatted.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
