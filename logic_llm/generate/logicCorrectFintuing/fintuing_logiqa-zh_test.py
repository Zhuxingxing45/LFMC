from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

device = "cuda" # the device to load the model onto
# #llama3_logic_correct_ez_v6
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v6/llama3_logic_original_hf_merged'

#llama3_logic_correct_ez_v9
model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v9/llama3_logic_original_hf_merged'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processed_data = []
idx = 0
with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/zh_test.json", 'r') as f:
    dataset = json.load(f)
    for example in tqdm(dataset):
        context = example['context']
        query = example['query']
        options = example['options']
        correct_option = example['correct_option']



        prompt = "给定以下背景信息：\n" + context + f"\n对于以下问题：{query}\n  A){options[0]}  B){options[1]} C){options[2]} D){options[3]}\n" + "请提供正确的选项与推理过程。一步一步来推理："

        messages = [
            {"role": "system", "content": "阅读下面一段文字，这段文字后面会有一个问题和ABCD四个选项，运用逻辑推理选出最合适的选项作为问题的答案。"},
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
            with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_correct_fintue_ez_v9/logiqa-zh_fintuing_test.json", 'w') as ft:
                json.dump(processed_data, ft, ensure_ascii=False, indent=4) 
                print(f"{len(processed_data)} new data have been generated.\n")
    
    with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_correct_fintue_ez_v9/logiqa-zh_fintuing_test.json", 'w') as ft:
        json.dump(processed_data, ft, ensure_ascii=False, indent=4) 
        print(f"{len(processed_data)} new data have been generated.\n")

