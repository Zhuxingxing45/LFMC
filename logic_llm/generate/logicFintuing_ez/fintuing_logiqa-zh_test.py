from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

device = "cuda" # the device to load the model onto

# model_path = "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_fintue/llama3_logic_original_hf_merged"
model_path = "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_base_ez/llama3_logic_original_hf_merged"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

"""
{"role": "system", "content": "You are a logician who can generate logical reasoning processes based on given premises and clonclusions."},
prompt = "Given the following premises:\n" + premises + f"\nWe can conclude the hypothesis '{conclusion}' is {label}.\n" + "Please provide the reasoning process to verify this conclusion."

"""

processed_data = []
idx = 0
with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_test.json", 'r') as f:
    dataset = json.load(f)
    for example in tqdm(dataset):
        context = example['context']
        query = example['query']
        options = example['options']
        correct_option = example['correct_option']



        prompt = prompt = "给定以下背景信息：\n" + context + f"\n对于以下问题：{query}\n  A){options[0]}  B){options[1]} C){options[2]} D){options[3]}\n" + "请提供正确的选项。"

        messages = [
            {"role": "system", "content": "你是一名逻辑学家。请根据给定的背景信息和问题从选项中选择正确的答案。"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #print(response)
        new_data = {
            "id" : idx,
            "context": example['context'],
            "query" : example['query'],
            "options":example['options'],
            "correct_option" : example['correct_option'],
            "generate_answer" : response
        }
        idx += 1
        processed_data.append(new_data)
        if idx % 50 == 0:
            # with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/Reclor_fintuing_dev.json", 'w') as ft:
            #     json.dump(processed_data, ft, ensure_ascii=False, indent=4)
            #     print(f"{len(processed_data)} new data have been generated.\n")

            with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/logiqa-zh_fintuing_test.json", 'w') as ft:
                json.dump(processed_data, ft, ensure_ascii=False, indent=4) 
                print(f"{len(processed_data)} new data have been generated.\n")
    
    with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/logiqa-zh_fintuing_test.json", 'w') as ft:
        json.dump(processed_data, ft, ensure_ascii=False, indent=4) 
        print(f"{len(processed_data)} new data have been generated.\n")
    

