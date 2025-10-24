import json

with open('Wrong_Inference/FOLIO_reasoning_path.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #513

with open('Wrong_Inference/LogiQA_v2_reasoning_path.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #3129

with open('Wrong_Inference/Reclor_reasoning_path_origin.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset))  #1206

with open('Wrong_Inference/Reclor_reasoning_path.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #331

with open('Wrong_Inference/logiqa-zh_reasoning_path_1.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #2271

with open('Wrong_Inference/logiqa-zh_reasoning_path_2.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #1063

