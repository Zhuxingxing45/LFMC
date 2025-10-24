import json
from tqdm import tqdm
import ast

with open('./logic_corrections/logiqa-zh_gpt-4_p2.json', 'r') as f:
    raw_dataset = json.load(f)

dataset = []

for example in tqdm(raw_dataset):
    context = example['context']
    query = example['query']
    options = example['options']
    # correct_option = example['correct_option']
    # if correct_option == 0:
    #     answer = 'A'
    # elif correct_option == 1:
    #     answer = 'B'
    # elif correct_option == 2:
    #     answer = 'C'
    # else:
    #     answer = 'D'
    reference = example['reference']
    generate_answer = example['generate_answer']
    
    prompt = "给定以下背景信息：\n" + context + f"\n对于以下问题：{query}\n  A){options[0]}  B){options[1]} C){options[2]} D){options[3]}\n" + "请提供正确的选项和推理过程。\n" + f"原始推理过程如下：\n {generate_answer}\n" + f"然而，正确选项是{ reference }。请识别并解释原始推理过程中的错误，然后纠正这些错误并提供修正后的最终答案。请在最后一行提供明确的选项。"

    data = {
                "conversation": [
                    {
                        "system":"给定一组前提，一个结论，以及一个验证该结论有效性的推理过程，任务是确定该推理过程是否正确。如果不正确，找出错误步骤，解释错误原因，并从错误步骤开始纠正原始解答。回答应尽可能简洁。",
                        "input": prompt,
                        "output":example['raw_logic_corrections'][0]
                    }
                ]
            }
    dataset.append(data)

with open('logiqa-zh_correct_fintuing_data_formatted_2.json', 'w', encoding = 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent = 4)
