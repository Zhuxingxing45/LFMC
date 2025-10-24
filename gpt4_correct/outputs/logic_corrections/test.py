import json

with open('FOLIO_gpt-4.json', 'r', encoding = 'utf-8') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #432

with open('Reclor_gpt-4.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #558    4-4082

with open('LogiQA_v2_gpt-4.json', 'r', encoding = 'utf-8') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #675   21-2434