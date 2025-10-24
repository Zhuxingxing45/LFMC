import json
import os
import re

def extract_option_label(text):
    # 使用正则表达式匹配选项标签
    match = re.match(r'([A-Z])\.?\s*\1\)\s*', text)
    if match:
        return match.group(1)
    return None

def WrongDataFilter(dataName) :
    wrong_dataset = []
    right_dataset = []

    with open(os.path.abspath(os.path.join('../data/logicotFintuing', dataName)), 'r') as f :
        file_content = f.read()  # 读取文件内容
        dataset = json.loads(file_content) 

    for data in dataset:
    #     if data['label'] in ['True', 'entailment', '0']:
    #         gold_answer = 'A'
    #     elif data['label'] in ['False', 'not-entailment', '1']:
    #         gold_answer = 'B'
    #     elif data['label'] in ['Uncertain','2']:
    #         gold_answer = 'C'
    #     else :
    #         gold_answer = 'D'
        predict_label = data['generate_answer'][0]

        if predict_label == data['reference'] :
            right_dataset.append(data)
        else :
            wrong_dataset.append(data)
    
    with open(os.path.join('../data/Wrong_Inference', dataName), 'w') as f:
        json.dump(wrong_dataset, f, indent=2, ensure_ascii=False)

    with open(os.path.join('../data/Rright_Inference', dataName), 'w') as f:
        json.dump(right_dataset, f, indent=2, ensure_ascii=False)


    
WrongDataFilter('FOLIO_reasoning_path.json')
WrongDataFilter('LogiQA_v2_reasoning_path.json')
WrongDataFilter('Reclor_reasoning_path.json')
    