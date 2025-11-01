import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Self, Dict, List, Tuple
from utils import OpenAIModel
import argparse

class LogicCorrection:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.prompt_creator = {'FOLIO': self.prompt_folio,
                               'LogiQA_v2':self.prompt_logiqa_v2,
                                'Reclor':self.prompt_reclor
                               }
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def prompt_folio(self, test_data):
        problem = test_data['premises']
        question = test_data['conclusion'].strip()
        original_reasoning = test_data['generate_answer']
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question).replace('[[ORIGINAL_REASONING]]',original_reasoning)
        return full_prompt

    def prompt_logiqa_v2(self, test_data):
        problem = test_data['premise']
        question = test_data['hypothesis'].strip()
        original_reasoning = test_data['generate_answer']
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question).replace('[[ORIGINAL_REASONING]]',original_reasoning)
        return full_prompt
    
    def prompt_reclor(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        answers = test_data['answers']
        original_reasoning = test_data['generate_answer']
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question).replace('[[ANSWERS]]', answers).replace('[[ORIGINAL_REASONING]]',original_reasoning)
        return full_prompt

    

    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, 'logicotFintuing', f'{self.dataset_name}_reasoning_path.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def logic_correction_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name}.")

        outputs = []
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
                outputs.append(output)

                
            except:
                print('Error in generating logic corrections for example.')

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''
    def batch_logic_correction_generation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name}.")

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
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
                        print('Error in generating logic programs for example')

        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    #parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_corrections')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LogicCorrection(args)
    logic_program_generator.batch_logic_correction_generation() 