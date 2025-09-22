from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/home/23_zxx/workspace/llama3-ft/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/home/23_zxx/workspace/llama3-ft/Meta-Llama-3-8B-Instruct")

"""
{"role": "system", "content": "You are a logician who can generate logical reasoning processes based on given premises and clonclusions."},
prompt = "Given the following premises:\n" + premises + f"\nWe can conclude the hypothesis '{conclusion}' is {label}.\n" + "Please provide the reasoning process to verify this conclusion."

"""

processed_data = []
idx = 0
with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/Reclor/val_500.json", 'r') as f:
    dataset = json.load(f)
    for example in tqdm(dataset):
        context = example['context']
        question = example['question']
        answers = example['answers']
        label = example['label']



        prompt = "Given the following context:\n" + context + f"\nFor the following question:{question}\n Which of the following options is correct? A){answers[0]}, B){answers[1]}, C){answers[2]}, D){answers[3]}\n" + "Please provide the correct option.There is no need to provide the reasoning process. " #"Please provide the correct option and the reasoning process to verify this conclusion, and indicate the correct option in the last sentence."

        # with open("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/icl_examples/Reclor_Llama.txt", 'r') as fi:
        #     fewshot = fi.read()


        # prompt = fewshot.replace("[[INPUT]]", prompt)
        messages = [
            {"role": "system", "content": "You are a logician. Please select the correct answer from the options based on the given context and question."},#, and provide the reasoning process
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
            "generate_answer" : response
        }
        idx += 1
        processed_data.append(new_data)
        if idx % 50 == 0:
            with open ("/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_answer_only/Reclor_baseline_dev.json", 'w') as ft:
                json.dump(processed_data, ft, ensure_ascii=False, indent=4)
                print(f"{len(processed_data)} new data have been generated.\n")
    

