#!/bin/bash
set -e 
set -x 

# ===============================
#  Step 1: Data Generation
# ===============================
echo "🔹 Step 1: Generating raw datasets..."

#data-format-coversation
python data/LogiQA_2/gdata_base.py
python data/Reclor/gdata_base.py
python data/FOLIO/gdata_base.py
python data/logiqa-zh/gdata_base.py

#data-format-instruction
python data/instruction_format_data/convert_all_to_alpaca.py

echo "Data generation complete."


# ===============================
#  Step 2: Fine-tuning Models
# ===============================
echo "Step 2: Fine-tuning models with LOCD..."

#Fine-tune LLaMA3-8B with LOCD
xtuner train configs/llama/llama3_correction/llama3_8b_instruct_qlora_locd.py --work-dir root/llama3/8b/llama3_logic_locd/llama3_logic_original_pth

#Fine-tune internLM2-7B with LOCD
xtuner train configs/internlm/internlm2-7b/internlm2_7b_qlora_locd.py --work-dir root/internlm2/7b/internlm2_logic_locd/internlm2_logic_original_pth

#Fine-tune Qwen3-4B with LOCD
xtuner train configs/qwen/qwen3_4b/qwen3_4b_qlora_locd.py --work-dir root/qwen3/4b/qwen3_logic_locd/qwen3_logic_original_pth

#Fine-tune Qwen3-8B with LOCD
xtuner train configs/qwen/qwen3_8b/qwen3_8b_qlora_locd.py --work-dir root/qwen3/8b/qwen3_logic_locd/qwen3_logic_original_pth

echo "Fine-tuning pth complete."

echo "Step 3: Converting and merging checkpoints..."

# ======================================================
# Convert Checkpoints (PTH → HuggingFace Adapter)
# ======================================================

# Example: InternLM2-7B
xtuner convert pth_to_hf \
    configs/internlm/internlm2-7b/internlm2_7b_qlora_locd.py \
    root/internlm2/7b/internlm2_logic_locd/internlm2_logic_original_pth/iter_6483.pth \
    root/internlm2/7b/internlm2_logic_locd/internlm2_logic_original_hf_adapter

# (Repeat as needed for other models if required)

# ======================================================
# Merge Adapters (HF Adapter + Base Model → Merged HF Model)
# ======================================================

export MKL_SERVICE_FORCE_INTEL=1

# Example: InternLM2-7B
xtuner convert merge \
    internlm2/internlm2-chat-7b \
    root/internlm2/7b/internlm2_logic_locd/internlm2_logic_original_hf_adapter \
    root/internlm2/7b/internlm2_logic_locd/internlm2_logic_original_hf_merged

echo "Conversion and merging complete!"
echo "All merged models are ready for evaluation."