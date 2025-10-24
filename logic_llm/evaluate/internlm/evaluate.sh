# #7b-origin
# #评估
python logic_llm/internlm/evaluate.py \
    --model_path root/internlm2/7b/internlm2_logic_origin/internlm2_logic_original_hf_merged \
    --output_path /internlm2/7b/origin_1 \
    --result_path /internlm2/7b/origin_1.json


# #7b-lfud
python logic_llm/internlm/evaluate.py \
    --model_path root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_hf_merged \
    --output_path /internlm2/7b/lfud_1 \
    --result_path /internlm2/7b/lfud_1.json


#7b-logic
python logic_llm/internlm/evaluate.py \
    --model_path root/internlm2/7b/internlm2_logic_correct_v1/internlm2_logic_original_hf_merged \
    --output_path /internlm2/7b/correct_v1_1 \
    --result_path /internlm2/7b/correct_v1_1.json

