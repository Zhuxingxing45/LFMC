import json
from tqdm import tqdm
import ast



with open('/home/23_zxx/project/LLM_correcting/outputs/logic_corrections/FOLIO_gpt-4.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    premises = example['premises']
    conclusion = example['conclusion']
    reference = example['reference']
    generate_answer = example['generate_answer']
    raw_logic_corrections = example['raw_logic_corrections']
 
    prompt = "Given the following premises:\n" + premises + f"\nFor the following hypothesis:{conclusion}\nWhich of the following options is correct? A)True, B)False, C)Uncertain \n" + "Please provide the correct option and the reasoning process to verify this conclusion.\n" + f"The original reasoning process is as follows:\n {generate_answer}\n" + f"However, the correct option is{reference}.Please identify and explain the mistakes in the original reasoning process, then correct these mistakes and provide the corrected final answer." 
    

    data = {
                "conversation": [
                    {
                        "system":"Given a set of premises, a conclusion, and a reasoning process for the validity of this conclusion, the task is to identify whether the reasoning process is correct. If it is not correct, find the erroneous steps, explain the reasons for the errors, and correct the original solution from the erroneous steps. The response should be as concise as possible.",
                        "input": prompt,
                        "output":raw_logic_corrections[0] # + f"\nTherefore, the answer is:{reference})."
                    }
                ]
            }
    dataset.append(data)

with open('FOLIO_correct_fintuing_data_formatted.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
