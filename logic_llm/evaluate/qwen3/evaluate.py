from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import json
from tqdm import tqdm
import os

from compute import save_em

# torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,default='root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged',  help="Path to model")
parser.add_argument("--output_path", type=str, default="/qwen3/8b/origin", help="Output JSON file")
parser.add_argument("--result_path", type=str, default="/qwen3/8b/origin.json", help="Result JSON file")
args = parser.parse_args()

device = 'cuda'

test_files = [
    'data/FOLIO/folio_v2_validation_203.jsonl',
    'data/LogiQA_v2/val_1k.jsonl',
    'data/logiqa-zh/zh_test.json',
    'data/Reclor/val_500.json'
]
dataset_names = {
        'FOLIO': 'FOLIO_fintuing_dev.json',
        'LogiQA_v2': 'LogiQA_v2_fintuing_dev.json',
        'Reclor': 'Reclor_fintuing_dev.json',
        'logiqa-zh':'logiqa-zh_fintuing_test.json'
    }

base_output_path = 'logic_llm/results'
output_path =base_output_path + args.output_path
os.makedirs(output_path, exist_ok=True)

model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

def generate_response(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256,  
        do_sample=True,   
        eos_token_id=tokenizer.eos_token_id, 
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return response

for file_path in test_files:
    processed_data = []
    idx = 0

    if "FOLIO" in file_path:
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Processing FOLIO")):
                data = json.loads(line)
                premises, conclusion, label = data['premises'], data['conclusion'], data['label']
                prompt = (
                    f"Given the following premises:\n{premises}\n"
                    f"For the following hypothesis:{conclusion}\n"
                    "Which of the following options is correct? A)True, B)False, C)Uncertain\n"
                    'Please only give the final answer (A/B/C).\nAnswer and reasoning step by step:'
                )
                messages = [
                    {"role": "system", "content": "You are a logician. Please select the correct answer from the options based on the given context and question."},
                    {"role": "user", "content": prompt}
                ]
                response = generate_response(messages)
                new_data = {
                    "folio_id": idx,
                    "premises": premises,
                    "conclusion": conclusion,
                    "label": label,
                    "generate_answer": response
                }
                idx += 1
                processed_data.append(new_data)

    elif "LogiQA_v2" in file_path:
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Processing LogiQA_v2")):
                data = json.loads(line)
                premise, hypothesis, label = data['premise'], data['hypothesis'], data['label']
                prompt = (
                    f"Given the following premises:\n{premise}\n"
                    f"For the following hypothesis:{hypothesis}\n"
                    "Which of the following options is correct? A)entailment, B)not-entailment\n"
                    'Please only give the final answer (A/B). \nAnswer and reasoning step by step:'
                )
                messages = [
                    {"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question and give the reasoning process. Please answer and reasoning step by step."},
                    {"role": "user", "content": prompt}
                ]
                response = generate_response(messages)
                new_data = {
                    "LogiQA_id": idx,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label,
                    "generate_answer": response
                }
                idx += 1
                processed_data.append(new_data)

    elif "logiqa-zh" in file_path:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
            for example in tqdm(dataset, desc="Processing logiqa-zh"):
                context, query, options, correct_option = example['context'], example['query'], example['options'], example['correct_option']
                prompt = (
                    f"给定以下背景信息：\n{context}\n"
                    f"对于以下问题：{query}\n"
                    f"A){options[0]}  B){options[1]}  C){options[2]}  D){options[3]}\n"
                    "请提供正确的选项与推理过程。一步一步来推理："
                )
                messages = [
                    {"role": "system", "content": "阅读下面一段文字，这段文字后面会有一个问题和ABCD四个选项，运用逻辑推理选出最合适的选项作为问题的答案。"},
                    {"role": "user", "content": prompt}
                ]
                response = generate_response(messages)
                new_data = {
                    "id": idx,
                    "context": context,
                    "query": query,
                    "options": options,
                    "correct_option": correct_option,
                    "generate_answer": response
                }
                idx += 1
                processed_data.append(new_data)

    elif "Reclor" in file_path:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
            for example in tqdm(dataset, desc="Processing Reclor"):
                context, question, answers, label = example['context'], example['question'], example['answers'], example['label']
                prompt = (
                    f"Given the following context:\n{context}\n"
                    f"For the following question:{question}\n"
                    f"Which of the following options is correct? A){answers[0]}, B){answers[1]}, C){answers[2]}, D){answers[3]}\n"
                    # 'Please only give the final answer (A/B/C/D), do not include any reasoning.'
                    "Please provide the correct option.\nAnswer and reasoning step by step:"
                )
                messages = [
                    {"role": "system", "content": "You are a logician. Please select the correct answer from the options based on the given context and question."},
                    {"role": "user", "content": prompt}
                ]
                response = generate_response(messages)
                new_data = {
                    "reclor_id": idx,
                    "context": context,
                    "question": question,
                    "answers": answers,
                    "label": label,
                    "generate_answer": response
                }
                idx += 1
                processed_data.append(new_data)

    dataset_key = None

    if "FOLIO" in file_path:
        dataset_key = "FOLIO"
    elif "LogiQA" in file_path:
        dataset_key = "LogiQA_v2"
    elif "logiqa-zh" in file_path:
        dataset_key = "logiqa-zh"
    elif "Reclor" in file_path:
        dataset_key = "Reclor"

    output_file = os.path.join(output_path, dataset_names[dataset_key])
    with open(output_file, 'w') as ft:
        json.dump(processed_data, ft, ensure_ascii=False, indent=4)
    print(f"{len(processed_data)} new data have been generated and saved to {output_file}\n")

result_base_url = 'logic_llm/evaluate/outputs'
result_path = result_base_url+args.result_path
save_em(output_path, result_path)
