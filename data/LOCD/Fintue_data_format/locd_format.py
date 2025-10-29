import json
from tqdm import tqdm
import os

data_dir = "LFMC/data/LOCD/"
output_dir = "LFMC/data/LOCD/"  

datasets_info = [
    {
        "input": "logic_corrections/FOLIO_gpt-4.json",
        "lang": "en",
        "output": "FOLIO_correct_finetuning_data_formatted.json"
    },
    {
        "input": "logic_corrections/LogiQA_v2_gpt-4.json",
        "lang": "en",
        "output": "LogiQA_correct_finetuning_data_formatted.json"
    },
      {
        "input": "logic_corrections/logiqa-zh_gpt-4.json",
        "lang": "zh",
        "output": "logiqa-zh_correct_finetuning_data_formatted.json"
    },
    {
        "input": "logic_corrections/logiqa-zh_gpt-4_p2.json",
        "lang": "zh",
        "output": "logiqa-zh_correct_finetuning_data_formatted_2.json"
    },
    {
        "input": "logic_corrections/Reclor_gpt-4.json",
        "lang": "en",
        "output": "Reclor_correct_finetuning_data_formatted.json"
    }
]

for info in datasets_info:
    input_path = os.path.join(data_dir, info["input"])
    output_path = os.path.join(output_dir, info["output"])
    lang = info["lang"]

    print(f"Processing {info['output']} ...")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)

    dataset = []

    for example in tqdm(raw_dataset):
        if "FOLIO" in info["output"]:
            premises = example['premises']
            conclusion = example['conclusion']
            reference = example['reference']
            generate_answer = example['generate_answer']
            raw_logic_corrections = example['raw_logic_corrections']

            prompt = (
                f"Given the following premises:\n{premises}\n"
                f"For the following hypothesis:{conclusion}\n"
                "Which of the following options is correct? A)True, B)False, C)Uncertain\n"
                "Please provide the correct option and the reasoning process to verify this conclusion.\n"
                f"The original reasoning process is as follows:\n {generate_answer}\n"
                f"However, the correct option is {reference}. Please identify and explain the mistakes in the original reasoning process, "
                "then correct these mistakes and provide the corrected final answer."
            )

        elif "LogiQA_v2" in info["output"]:
            premise = example['premise']
            hypothesis = example['hypothesis']
            reference = example['reference']
            generate_answer = example['generate_answer']

            prompt = (
                f"Given the following premises:\n{premise}\n"
                f"For the following hypothesis:{hypothesis}\n"
                "Which of the following options is correct? A)entailment, B)not-entailment\n"
                "Please provide the correct option and the reasoning process to verify this conclusion.\n"
                f"The original reasoning process is as follows:\n {generate_answer}\n"
                f"However, the correct option is {reference}. Please identify and explain the mistakes in the original reasoning process, "
                "then correct these mistakes and provide the corrected final answer. Please provide the explicit option in the final line."
            )

        elif "logiqa-zh" in info["output"]:
            context = example['context']
            query = example['query']
            options = example['options']
            reference = example['reference']
            generate_answer = example['generate_answer']

            prompt = (
                f"给定以下背景信息：\n{context}\n"
                f"对于以下问题：{query}\n"
                f"  A){options[0]}  B){options[1]}  C){options[2]}  D){options[3]}\n"
                "请提供正确的选项和推理过程。\n"
                f"原始推理过程如下：\n {generate_answer}\n"
                f"然而，正确选项是{reference}。请识别并解释原始推理过程中的错误，然后纠正这些错误并提供修正后的最终答案。"
                "请在最后一行提供明确的选项。"
            )

        elif "Reclor" in info["output"]:
            context = example['context']
            question = example['question']
            answers = example['answers']
            reference = example['reference']
            generate_answer = example['generate_answer']

            prompt = (
                f"Given the following context:\n{context}\n"
                f"For the following question:{question}\n"
                f"Which of the following options is correct? A){answers[0]}, B){answers[1]}, C){answers[2]}, D){answers[3]}\n"
                "Please provide the correct option and the reasoning process to verify this conclusion.\n"
                f"The original reasoning process is as follows:\n {generate_answer}\n"
                f"However, the correct option is {reference}. Please identify and explain the mistakes in the original reasoning process, "
                "then correct these mistakes and provide the corrected final answer. Please provide the explicit option in the final line."
            )

        system_prompt = (
            "Given a set of premises, a conclusion, and a reasoning process for the validity of this conclusion, "
            "the task is to identify whether the reasoning process is correct. If it is not correct, find the erroneous steps, "
            "explain the reasons for the errors, and correct the original solution from the erroneous steps. "
            "The response should be as concise as possible."
            if lang == "en"
            else "给定一组前提，一个结论，以及一个验证该结论有效性的推理过程，任务是确定该推理过程是否正确。如果不正确，找出错误步骤，解释错误原因，并从错误步骤开始纠正原始解答。回答应尽可能简洁。"
        )

        data = {
            "conversation": [
                {
                    "system": system_prompt,
                    "input": prompt,
                    "output": example['raw_logic_corrections'][0]
                }
            ]
        }
        dataset.append(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(dataset)} samples to {output_path}\n")
