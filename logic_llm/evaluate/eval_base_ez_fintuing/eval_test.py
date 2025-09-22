# import json
# from sklearn.metrics import f1_score
# from tqdm import tqdm
# import re

# def compute_em(predictions, references):
#     """Compute Exact Match (EM) score."""
#     return sum([int(pred == ref) for pred, ref in zip(predictions, references)]) / len(predictions)

# def compute_f1(predictions, references):
#     """Compute F1 score."""
#     f1_scores = [f1_score([ref], [pred], average='macro') for pred, ref in zip(predictions, references)]
#     return sum(f1_scores) / len(f1_scores)

# def load_dataset(file_path):
#     """Load dataset from a JSON file."""
#     with open(file_path, 'r') as file:
#         dataset = json.load(file)
    
#     predictions = []
#     references = []
#     if file_path == '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/FOLIO_fintuing_dev.json':
#         with open(file_path, 'r') as file:
#             dataset = json.load(file)
#         for data in tqdm(dataset):
#             label = data['label']
#             if label == 'True':
#                 reference = 'A'
#             elif label == 'False':
#                 reference = 'B'
#             elif label == 'Uncertain':
#                 reference = 'C'
#             references.append(reference)

#             generate_answer = data['generate_answer']
#             match = re.search(r'\b[A-D]\)', generate_answer)
#             if match:
#                 # print(match)                
#                 correct_option = match.group(0)
#                 # print(f'The correct option is: {correct_option}')
#             else:
#                 correct_option = ""
#             #print(correct_option)
#             prediction = correct_option.strip(')')[0] if correct_option != "" else ""
#             predictions.append(prediction)
#     elif file_path == '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/LogiQA_v2_fintuing_dev.json':
#         with open(file_path, 'r') as file:
#             dataset = json.load(file)
#         for data in tqdm(dataset):
#             label = data['label']
#             if label == 'entailment':
#                 reference = 'A'
#             elif label == 'not-entailment':
#                 reference = 'B'
#             references.append(reference)

#             generate_answer = data['generate_answer']
#             match = re.search(r'\b[A-D]\)', generate_answer)
#             if match:
#                 # print(match)                
#                 correct_option = match.group(0)
#                 # print(f'The correct option is: {correct_option}')
#             else:
#                 correct_option = ""
#             #print(correct_option)
#             prediction = correct_option.strip(')')[0] if correct_option != "" else ""
#             predictions.append(prediction)
#     elif file_path == '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/Reclor_fintuing_dev.json':
#         with open(file_path, 'r') as file:
#             dataset = json.load(file)
#         for data in tqdm(dataset):
#             label = data['label']
#             if label == 0:
#                 reference = 'A'
#             elif label == 1:
#                 reference = 'B'
#             elif label == 2:
#                 reference = 'C'
#             elif label == 3:
#                 reference = 'D'
#             references.append(reference)

#             generate_answer = data['generate_answer']
#             match = re.search(r'\b[A-D]\)', generate_answer)
#             if match:
#                 # print(match)                
#                 correct_option = match.group(0)
#                 # print(f'The correct option is: {correct_option}')
#             else:
#                 correct_option = ""
#             #print(correct_option)
#             prediction = correct_option.strip(')')[0] if correct_option != "" else ""
#             predictions.append(prediction)

#     elif file_path == '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/logiqa-zh_fintuing_test.json':
#         with open(file_path, 'r') as file:
#             dataset = json.load(file)
#         for data in tqdm(dataset):
#             correct_option = data['correct_option']
#             if correct_option == 0:
#                 reference = 'A'
#             elif correct_option == 1:
#                 reference = 'B'
#             elif correct_option == 2:
#                 reference = 'C'
#             elif correct_option == 3:
#                 reference = 'D'
#             references.append(reference)

#             generate_answer = data['generate_answer']
#             match = re.search(r'\b[A-D]\)', generate_answer)
#             if match:
#                 # print(match)                
#                 correct_option = match.group(0)
#                 # print(f'The correct option is: {correct_option}')
#             else:
#                 correct_option = ""
#             #print(correct_option)
#             prediction = correct_option.strip(')')[0] if correct_option != "" else ""
#             predictions.append(prediction)
            
#     return predictions, references

# def evaluate_datasets(dataset_paths):
#     results = {}
#     for dataset_name, dataset_path in dataset_paths.items():
#         predictions, references = load_dataset(dataset_path)
#         em = compute_em(predictions, references)
#         f1 = compute_f1(predictions, references)
#         results[dataset_name] = {'EM': em, 'F1': f1}
#     return results

# def write_results_to_file(results, file_path):
#     """Write evaluation results to a JSON file."""
#     with open(file_path, 'w') as file:
#         json.dump(results, file, indent=4)

# def main():
#     # 数据集路径
#     dataset_paths = {
#         'FOLIO': '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/FOLIO_fintuing_dev.json',
#         'LogiQA_v2': '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/LogiQA_v2_fintuing_dev.json',
#         'Reclor': '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/Reclor_fintuing_dev.json',
#         'logiqa-zh':'/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez/logiqa-zh_fintuing_test.json'
#     }

#     # 评估数据集
#     results = evaluate_datasets(dataset_paths)

#     # 记录评估结果
#     results_file_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/evaluate/outputs/logic_base_fintue_ez_evaluation_results.json'
#     write_results_to_file(results, results_file_path)

#     print(f"Evaluation results written to {results_file_path}")

# if __name__ == "__main__":
#     main()


import json
from sklearn.metrics import f1_score
from tqdm import tqdm
import re
import os

def compute_em(predictions, references):
    """Compute Exact Match (EM) score."""
    return sum([int(pred == ref) for pred, ref in zip(predictions, references)]) / len(predictions)

def compute_f1(predictions, references):
    """Compute F1 score."""
    f1_scores = [f1_score([ref], [pred], average='macro') for pred, ref in zip(predictions, references)]
    return sum(f1_scores) / len(f1_scores)

def load_dataset(file_path, file_name):
    # """Load dataset from a JSON file."""
    # with open(file_path, 'r') as file:
    #     dataset = json.load(file)
    
    predictions = []
    references = []
    if file_name == 'FOLIO_fintuing_dev.json':
    # if file_name == 'FOLIO_baseline_dev_v2.json':

        with open(os.path.join(file_path,file_name), 'r') as file:
            dataset = json.load(file)
        for data in tqdm(dataset):
            label = data['label']
            if label == 'True':
                reference = 'A'
            elif label == 'False':
                reference = 'B'
            elif label == 'Uncertain':
                reference = 'C'
            references.append(reference)

            generate_answer = data['generate_answer']
            match = re.search(r'\b[A-D]\)', generate_answer)
            if match:
                # print(match)                
                correct_option = match.group(0)
                # print(f'The correct option is: {correct_option}')
            else:
                correct_option = ""
            #print(correct_option)
            prediction = correct_option.strip(')')[0] if correct_option != "" else ""
            predictions.append(prediction)
    elif file_name == 'LogiQA_v2_fintuing_dev.json':
    # elif file_name == 'LogiQA_v2_baseline_dev.json':
        with open(os.path.join(file_path,file_name), 'r') as file:
            dataset = json.load(file)
        for data in tqdm(dataset):
            label = data['label']
            if label == 'entailment':
                reference = 'A'
            elif label == 'not-entailment':
                reference = 'B'
            references.append(reference)

            generate_answer = data['generate_answer']
            match = re.search(r'\b[A-D]\)', generate_answer)
            if match:
                # print(match)                
                correct_option = match.group(0)
                # print(f'The correct option is: {correct_option}')
            else:
                correct_option = ""
            #print(correct_option)
            prediction = correct_option.strip(')')[0] if correct_option != "" else ""
            predictions.append(prediction)
    elif file_name == 'Reclor_fintuing_dev.json':
    # elif file_name == 'Reclor_baseline_dev.json':
        with open(os.path.join(file_path,file_name), 'r') as file:
            dataset = json.load(file)
        for data in tqdm(dataset):
            label = data['label']
            if label == 0:
                reference = 'A'
            elif label == 1:
                reference = 'B'
            elif label == 2:
                reference = 'C'
            elif label == 3:
                reference = 'D'
            references.append(reference)

            generate_answer = data['generate_answer']
            match = re.search(r'\b[A-D]\)', generate_answer)
            if match:
                # print(match)                
                correct_option = match.group(0)
                # print(f'The correct option is: {correct_option}')
            else:
                correct_option = ""
            #print(correct_option)
            prediction = correct_option.strip(')')[0] if correct_option != "" else ""
            predictions.append(prediction)

    elif file_name == 'logiqa-zh_fintuing_test.json':
    # elif file_name == 'logiqa-zh_baseline_test.json':
        with open(os.path.join(file_path,file_name), 'r') as file:
            dataset = json.load(file)
        for data in tqdm(dataset):
            correct_option = data['correct_option']
            if correct_option == 0:
                reference = 'A'
            elif correct_option == 1:
                reference = 'B'
            elif correct_option == 2:
                reference = 'C'
            elif correct_option == 3:
                reference = 'D'
            references.append(reference)

            generate_answer = data['generate_answer']
            match = re.search(r'\b[A-D]\)', generate_answer)
            if match:
                # print(match)                
                correct_option = match.group(0)
                # print(f'The correct option is: {correct_option}')
            else:
                correct_option = ""
            #print(correct_option)
            prediction = correct_option.strip(')')[0] if correct_option != "" else ""
            predictions.append(prediction)
            
    return predictions, references

def evaluate_datasets(file_path, dataset_names):
    results = {}
    for dataset, dataset_name in dataset_names.items():
        predictions, references = load_dataset(file_path, dataset_name)
        em = compute_em(predictions, references)
        # f1 = compute_f1(predictions, references)
        results[dataset] = {'EM': em}
    return results

def write_results_to_file(results, file_path):
    """Write evaluation results to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

def main():
    # 数据集路径  
    file_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_fintue_ez'
    # file_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/results/logic_base_answer_only'

    dataset_names = {
        'FOLIO': 'FOLIO_fintuing_dev.json',
        'LogiQA_v2': 'LogiQA_v2_fintuing_dev.json',
        'Reclor': 'Reclor_fintuing_dev.json',
        'logiqa-zh':'logiqa-zh_fintuing_test.json'
    }

    # dataset_names = {
    #     'FOLIO': 'FOLIO_baseline_dev_v2.json',
    #     'LogiQA_v2': 'LogiQA_v2_baseline_dev.json',
    #     'Reclor': 'Reclor_baseline_dev.json',
    #     'logiqa-zh':'logiqa-zh_baseline_test.json'
    # }

    # 评估数据集
    results = evaluate_datasets(file_path, dataset_names)

    # 记录评估结果
    results_file_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/evaluate/outputs/logic_base_fintue_ez_evaluation_results.json'
    # results_file_path = '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/logic_llm/evaluate/outputs/logic_base_answer_only_evaluation_results.json'

    write_results_to_file(results, results_file_path)

    print(f"Evaluation results written to {results_file_path}")

if __name__ == "__main__":
    main()
