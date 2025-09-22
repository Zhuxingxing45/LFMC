from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

device = "cuda" # the device to load the model onto

# #llama3_logic_correct_ez_FOLIO
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_FOLIO/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v2
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v2/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v3
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v3/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v4
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v4/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v5
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v5/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v6
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v6/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v8
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v8/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v9
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v9/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v10
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v10/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v11
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v11/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v12
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v12/llama3_logic_original_hf_merged'

#llama3_logic_correct_ez_v13
model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v13/llama3_logic_original_hf_merged'

# #llama3_logic_correct_ez_v14
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v14/llama3_logic_original_hf_merged'

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
with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/FOLIO/folio_v2_validation_203.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        premises = data['premises']
        conclusion = data['conclusion']
        label = data['label']
        #prompt = "Given the following premises:\n" + premises + f"\nWe can conclude the hypothesis '{conclusion}' is {label}.\n" + "Please provide the reasoning process to verify this conclusion."
        prompt = "Given the following premises:\n" + premises + f"\nFor the following hypothesis:{conclusion}\nWhich of the following options is correct? A)True, B)False, C)Uncertain \n" + "Please provide the correct option. "

        messages = [
            {"role": "system", "content": "You are a logician. Please select the correct answer from the options based on the given context and question."},
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
        new_data = {
            "folio_id" : idx,
            "premises": data['premises'],
            "conclusion" : data['conclusion'],
            "label" : data['label'],
            "generate_answer" : response
        }
        idx += 1
        processed_data.append(new_data)
        if idx % 50 == 0:
            with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_correct_fintue_ez_v13/FOLIO_fintuing_dev.json", 'w') as ft:
                json.dump(processed_data, ft, ensure_ascii=False, indent=4)
                print(f"{len(processed_data)} new data have been generated.\n")


    with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_correct_fintue_ez_v13/FOLIO_fintuing_dev.json", 'w') as ft:
        json.dump(processed_data, ft, ensure_ascii=False, indent=4)
        print(f"{len(processed_data)} new data have been generated.\n")        
    
    

