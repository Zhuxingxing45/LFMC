from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

device = "cuda" # the device to load the model onto

# #llama3_logic_logicot_v2
# model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_logicot_v2/llama3_logic_logicot_hf_merged'

#llama3_logic_correct_ez_v14
model_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_correct_ez_v14/llama3_logic_original_hf_merged'


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

processed_data = []
idx = 0
with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/LogiQA_v2/val_1k.jsonl", 'r' ) as f:
    for line in f:
        data = json.loads(line)
        premise = data['premise']
        hypothesis = data['hypothesis']
        label = data['label']
        prompt = "Given the following premises:\n" + premise + f"\nFor the following hypothesis:{hypothesis}\nWhich of the following options is correct? A)entailment, B)not-entailment\n" + "Please provide the correct option. \nAnswer and reasoning step by step:"

        messages = [
            {"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question and give the reasoning process. Please answer and reasoning step by step."},
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
            "LogiQA_id" : idx,
            "premise": data['premise'],
            "hypothesis" : data['hypothesis'],
            "label" : data['label'],
            "generate_answer" : response
        }
        idx += 1
        processed_data.append(new_data)
        if idx % 50 == 0:
            # with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/LogiQA_v2_fintuing_dev.json", 'w') as ft:
            #     json.dump(processed_data, ft, ensure_ascii=False, indent=4)
            #     print(f"{len(processed_data)} new data have been generated.\n")

            with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_logicot_fintue_v2/LogiQA_v2_fintuing_dev.json", 'w') as ft:
                json.dump(processed_data, ft, ensure_ascii=False, indent=4)
                print(f"{len(processed_data)} new data have been generated.\n")
    
    with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_logicot_fintue_v2/LogiQA_v2_fintuing_dev.json", 'w') as ft:
        json.dump(processed_data, ft, ensure_ascii=False, indent=4)
        print(f"{len(processed_data)} new data have been generated.\n")

