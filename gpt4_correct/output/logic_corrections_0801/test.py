import json

with open('FOLIO_gpt-4.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #432



with open('Reclor_gpt-4.json', 'r') as f:
    raw_dataset = json.load(f)
    print(len(raw_dataset)) #558    4-4082



