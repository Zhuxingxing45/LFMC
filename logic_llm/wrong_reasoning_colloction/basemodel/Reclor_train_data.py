from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

device = "cuda" # the device to load the model onto

# model_path = "/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_fintue/llama3_logic_original_hf_merged"
model_path = "/home/23_zxx/workspace/llama3-ft/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

processed_data = []
idx = 0
with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/Reclor/train_4.64k.json", 'r') as f:
    dataset = json.load(f)
    for example in tqdm(dataset):
        context = example['context']
        question = example['question']
        answers = example['answers']
        label = example['label']
        if label == 0:
            reference = 'A'
        elif label == 1:
            reference = 'B'
        elif label == 2:
            reference = 'C'
        elif label == 3:
            reference = 'D'


        prompt = "Given the following context:\n" + context + f"\nFor the following question:{question}\n Which of the following options is correct? A){answers[0]}, B){answers[1]}, C){answers[2]}, D){answers[3]}\n" + "Please provide the correct option and the reasoning process to verify this conclusion."

        messages = [
            {"role": "system", "content": "You are a logician , Please select the correct answer from the options based on the given context and question, and provide the reasoning process."},
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
            "reclor_id" : idx,
            "context": example['context'],
            "question" : example['question'],
            "answers":example['answers'],
            "label" : example['label'],
            "reference":reference,
            "generate_answer" : response
        }
        idx += 1
        processed_data.append(new_data)
        if idx % 10 == 0:

            with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/reasoning_colloction/output/basemodel/Reclor_reasoning_path.json", 'w') as ft:
                json.dump(processed_data, ft, ensure_ascii=False, indent=4)
                print(f"{len(processed_data)} new data have been generated.\n")
    

