

# #8b-origin
# #评估
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge \
  /home/23_zxx/workspace/huggingface/Qwen/Qwen3-8B \
  root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_adapter \
  root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/origin_1 \
    --result_path /qwen3/8b/origin_1.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/origin_2 \
    --result_path /qwen3/8b/origin_2.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/origin_3 \
    --result_path /qwen3/8b/origin_3.json
# #删除模型
rm -rf root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged

#8b-lfud
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /home/23_zxx/workspace/huggingface/Qwen/Qwen3-8B root/qwen3/8b/qwen3_logic_lfud/qwen3_logic_original_hf_adapter root/qwen3/8b/qwen3_logic_lfud/qwen3_logic_original_hf_merged

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/lfud_1 \
    --result_path /qwen3/8b/lfud_1.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/lfud_2 \
    --result_path /qwen3/8b/lfud_2.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/lfud_3 \
    --result_path /qwen3/8b/lfud_3.json

# rm -rf root/qwen3/8b/qwen3_logic_lfud/qwen3_logic_original_hf_merged

#8b-logic
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /home/23_zxx/workspace/huggingface/Qwen/Qwen3-8B root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_hf_adapter root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/correct_v1_1 \
    --result_path /qwen3/8b/correct_v1_1.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/correct_v1_2 \
    --result_path /qwen3/8b/correct_v1_2.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged \
    --output_path /qwen3/8b/correct_v1_3 \
    --result_path /qwen3/8b/correct_v1_3.json

# rm -rf root/qwen3/8b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged

#4b-origin
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /home/23_zxx/workspace/huggingface/Qwen/Qwen3-4B-Thinking-2507 root/qwen3/4b/qwen3_logic_origin/qwen3_logic_original_hf_adapter root/qwen3/4b/qwen3_logic_origin/qwen3_logic_original_hf_merged


python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/origin_1 \
    --result_path /qwen3/4b/origin_1.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/origin_2 \
    --result_path /qwen3/4b/origin_2.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_origin/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/origin_3 \
    --result_path /qwen3/4b/origin_3.json

# rm -rf root/qwen3/4b/qwen3_logic_origin/qwen3_logic_original_hf_merged

#4b-lfud
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /home/23_zxx/workspace/huggingface/Qwen/Qwen3-4B-Thinking-2507 root/qwen3/4b/qwen3_logic_lfud/qwen3_logic_original_hf_adapter root/qwen3/4b/qwen3_logic_lfud/qwen3_logic_original_hf_merged

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_lfud/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/lfud_1 \
    --result_path /qwen3/4b/lfud_1.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_lfud/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/lfud_2 \
    --result_path /qwen3/4b/lfud_2.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_lfud/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/lfud_3 \
    --result_path /qwen3/4b/lfud_3.json

# rm -rf root/qwen3/4b/qwen3_logic_lfud/qwen3_logic_original_hf_merged

#4b-logic
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /home/23_zxx/workspace/huggingface/Qwen/Qwen3-4B-Thinking-2507 root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_hf_adapter root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/correct_v1_1 \
    --result_path /qwen3/4b/correct_v1_1.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/correct_v1_2 \
    --result_path /qwen3/4b/correct_v1_2.json

python logic_llm/qwen3/evaluate.py \
    --model_path root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged \
    --output_path /qwen3/4b/correct_v1_3 \
    --result_path /qwen3/4b/correct_v1_3.json

# rm -rf root/qwen3/4b/qwen3_logic_correct_v1/qwen3_logic_original_hf_merged