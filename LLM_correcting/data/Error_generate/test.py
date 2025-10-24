import json

with open('/home/23_zxx/project/LLM_correcting/data/Error_generate/Reclor_reasoning_path_First.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #802

with open('/home/23_zxx/project/LLM_correcting/data/Error_generate/Reclor_reasoning_path.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #179


