import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import  Dict, List, Tuple
from utils import OpenAIModel
import argparse

class LogicCorrection:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.prompt_creator = {'FOLIO': self.prompt_folio,
                               'LogiQA_v2':self.prompt_logiqa_v2,
                                'Reclor':self.prompt_reclor,
                                'logiqa-zh':self.prompt_logiqa_zh
                               }

    def prompt_folio(self, test_data):
        premises = test_data['premises']
        conclusion = test_data['conclusion'].strip()
        generate_answer = test_data['generate_answer']
        reference = test_data['reference']
        # full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question).replace('[[ORIGINAL_REASONING]]',original_reasoning)
        full_prompt = "Given the following premises:\n" + premises + f"\nFor the following hypothesis:{conclusion}\nWhich of the following options is correct? A)True, B)False, C)Uncertain \n" + "Please provide the correct option and the reasoning process to verify this conclusion.\n" + f"The original reasoning process is as follows:\n {generate_answer}\n" + f"However, the correct option is{reference}.Please identify the logical mistakes in the original reasoning process and explain, then correct these mistakes and provide the corrected final answer.Please provide the explicit option in the final line." 
        return full_prompt

    def prompt_logiqa_v2(self, test_data):
        premise = test_data['premise']
        hypothesis = test_data['hypothesis'].strip()
        generate_answer = test_data['generate_answer']
        reference = test_data['reference']
        full_prompt = "Given the following premises:\n" + premise + f"\nFor the following hypothesis:{hypothesis}\nWhich of the following options is correct? A)entailment, B)not-entailment\n" + "Please provide the correct option and the reasoning process to verify this conclusion.\n" + f"The original reasoning process is as follows:\n {generate_answer}\n" + f"However, the correct option is{reference}.Please identify the logical mistakes in the original reasoning process  and explain, then correct these mistakes and provide the corrected final answer.Please provide the explicit option in the final line." 
        return full_prompt
    
    def prompt_reclor(self, test_data):
        context = test_data['context']
        question = test_data['question'].strip()
        answers = test_data['answers']
        generate_answer = test_data['generate_answer']
        reference = test_data['reference']
        full_prompt = "Given the following context:\n" + context + f"\nFor the following question:{question}\n Which of the following options is correct? A){answers[0]}, B){answers[1]}, C){answers[2]}, D){answers[3]}\n" + "Please provide the correct option and the reasoning process to verify this conclusion.\n" + f"The original reasoning process is as follows:\n {generate_answer}\n" + f"However, the correct option is{reference}.Please identify the logical mistakes in the original reasoning process and explain, then correct these mistakes and provide the corrected final answer.Please provide the explicit option in the final line."
        return full_prompt
    
    def prompt_logiqa_zh(self, test_data):
        context = test_data['context']
        query = test_data['query']
        options = test_data['options']
        # correct_option = test_data['correct_option']
        generate_answer = test_data['generate_answer']
        reference = test_data['reference']

        full_prompt = "给定以下背景信息：\n" + context + f"\n对于以下问题：{query}\n  A){options[0]}  B){options[1]} C){options[2]} D){options[3]}\n" + "请提供正确的选项和推理过程。\n" + f"原始推理过程如下：\n {generate_answer}\n" + f"然而，正确选项是{ reference }。请识别原始推理过程中的逻辑错误并解释，然后纠正这些错误并提供修正后的最终答案。请在最后一行提供明确的选项。"
        return full_prompt

    

    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, f'{self.dataset_name}_reasoning_path_2.json'), 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def logic_correction_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name}.")

        outputs = []
        error_generating_datas = []
        for example in tqdm(raw_dataset):
            # create prompt
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                output = self.openai_api.generate(full_prompt)
                # print(full_prompt)
                # programs = [output]
                corrections = [output]

                # create output
                if self.dataset_name == 'FOLIO':
                    output = {'id': example['folio_id'], 
                        'premises': example['premises'],
                        'conclusion': example['conclusion'], 
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                elif self.dataset_name == 'LogiQA_v2':
                    output = {'id': example['LogiQA_id'], 
                        'premise': example['premise'],
                        'hypothesis': example['hypothesis'], 
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                elif self.dataset_name == 'Reclor':
                    output = {'id': example['reclor_id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answers':example['answers'],
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                elif self.dataset_name == 'logiqa-zh':
                    output = {'id': example['id'], 
                        'context': example['context'],
                        'query': example['query'], 
                        'options':example['options'],
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                outputs.append(output)
                
            except:
                error_generating_datas.append(example)
                if len(error_generating_datas) %  10== 0:
                    with open(os.path.join(r'C:\Users\My\Desktop\LLM_correcting\data\Error_generate', f'{self.dataset_name}_reasoning_path.json'), 'w', encoding='utf-8') as f:
                        json.dump(error_generating_datas, f, indent=2, ensure_ascii=False)
                print('Error in generating logic corrections for example.')
            
            # save outputs
            if len(outputs) % 10 == 0:        
                with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)
            print(f"Generated {len(outputs)} examples.")
            

            
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''
    def batch_logic_correction_generation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name}.")

        outputs = []
        error_generating_datas = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks[29:]):
            # create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts)
                # create output
                for example, output in zip(chunk, batch_outputs):
                    corrections = [output]
                # create output
                if self.dataset_name == 'FOLIO':
                    output = {'id': example['folio_id'], 
                        'premises': example['premises'],
                        'conclusion': example['conclusion'], 
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                elif self.dataset_name == 'LogiQA_v2':
                    output = {'id': example['LogiQA_id'], 
                        'premise': example['premise'],
                        'hypothesis': example['hypothesis'], 
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                elif self.dataset_name == 'Reclor':
                    output = {'id': example['reclor_id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answers':example['answers'],
                        'reference': example['reference'],
                        'generate_answer': example['generate_answer'],
                        'raw_logic_corrections': corrections}
                outputs.append(output)
            except:
                # generate one by one if batch generation fails
                for example, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.openai_api.generate(full_prompt)
                        corrections = [output]
                        # create output
                        if self.dataset_name == 'FOLIO':
                            output = {'id': example['folio_id'], 
                                'premises': example['premises'],
                                'conclusion': example['conclusion'], 
                                'reference': example['reference'],
                                'generate_answer': example['generate_answer'],
                                'raw_logic_corrections': corrections}
                        elif self.dataset_name == 'LogiQA_v2':
                            output = {'id': example['LogiQA_id'], 
                                'premise': example['premise'],
                                'hypothesis': example['hypothesis'], 
                                'reference': example['reference'],
                                'generate_answer': example['generate_answer'],
                                'raw_logic_corrections': corrections}
                        elif self.dataset_name == 'Reclor':
                            output = {'id': example['reclor_id'], 
                                'context': example['context'],
                                'question': example['question'], 
                                'answers':example['answers'],
                                'reference': example['reference'],
                                'generate_answer': example['generate_answer'],
                                'raw_logic_corrections': corrections}
                        outputs.append(output)
                    except:
                        error_generating_datas.append(example)
                        with open(os.path.join(r'C:\Users\My\Desktop\LLM_correcting\data\Error_generate', f'{self.dataset_name}_reasoning_path.json'), 'w', encoding='utf-8') as f:
                            json.dump(error_generating_datas, f, indent=2, ensure_ascii=False)
                        print('Error in generating logic programs for example')

            # remove examples with duplicate ids from the result
            outputs = list({output['id']: output for output in outputs}.values())
            print(f"Generated {len(outputs)} examples.")
        
            # save outputs
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'C:\Users\My\Desktop\LLM_correcting\data\Wrong_Inference')
    parser.add_argument('--dataset_name', type=str)
    #parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default=r'C:\Users\My\Desktop\LLM_correcting\outputs\logic_corrections')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LogicCorrection(args)
    logic_program_generator.logic_correction_generation() 
