#Fine-tune LLaMA3-8B with LOCD
xtuner train configs/llama/llama3_correction/llama3_8b_instruct_qlora_logic_correct_ez.py --work-dir root/llama3/8b/llama3_logic_correct_v1/llama3_logic_original_pth

#Fine-tune internLM2-7B with LOCD
xtuner train configs/internlm/internlm2-7b/internlm2_7b_qlora_logic_e3.py --work-dir root/internlm2/7b/internlm2_logic_correct_v1/internlm2_logic_original_pth

#Fine-tune Qwen3-4B with LOCD
xtuner train configs/qwen/qwen3_4b/qwen3_4b_qlora_logic.py --work-dir root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_pth

#Fine-tune Qwen3-8B with LOCD
xtuner train configs/qwen/qwen3_8b/qwen3_8b_qlora_logic.py --work-dir root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_pth