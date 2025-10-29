#!/bin/bash
# ======================================================
# LFMC Dataset Preparation Script
# ======================================================
# 1. Download raw datasets from Hugging Face
# 2. Generate reasoning data with logical errors (via LLaMA3-8B)
# 3. Filter erroneous samples
# 4. Correct reasoning using GPT-4
# 5. Reformat data for fine-tuning
# ======================================================

set -e  # Exit immediately if a command exits with a non-zero status.

# ------------------------------
# Configurations
# ------------------------------
DATA_ROOT="data"
HF_CMD="huggingface-cli download --resume-download --repo-type dataset"
API_KEY=${OPENAI_API_KEY:-"<YOUR_OPENAI_API_KEY>"}  # Support environment variable

# Create base directory if not exists
mkdir -p "$DATA_ROOT"

# ------------------------------
# Step 1. Download Original Datasets
# ------------------------------
echo "Downloading origin datasets from Hugging Face..."

declare -A DATASETS=(
  ["tasksource/folio"]="tasksource/folio"
  ["tasksource/logiqa-2.0-nli"]="tasksource/logiqa-2.0-nli"
  ["jiacheng-ye/logiqa-zh"]="jiacheng-ye/logiqa-zh"
  ["metaeval/reclor"]= "metaeval/reclor"
)

for REPO in "${!DATASETS[@]}"; do
  LOCAL_DIR="${DATASETS[$REPO]}"
  echo "➡ Downloading $REPO..."
  if [ ! -d "$LOCAL_DIR" ]; then
      $HF_CMD "$REPO" --local-dir "$LOCAL_DIR"
  else
      echo "$LOCAL_DIR already exists, skipping..."
  fi
done

# ------------------------------
# Step 2. Generate Reasoning Data with Logical Errors
# ------------------------------
echo "Generating reasoning data with LLaMA3-8B..."

cd logic_llm/wrong_reasoning_colloction/basemodel

for FILE in FOLIO_train_data.py LogiQA_v2_train_data.py Reclor_train_data.py logiqa-zh_train_data.py; do
    echo "➡ Running $FILE ..."
    if ! python "$FILE"; then
        echo " $FILE 执行失败"
        exit 1
    fi
done

cd ../../../../

# ------------------------------
# Step 3. Filter Erroneous Reasoning Samples
# ------------------------------
echo "Filtering samples with logical errors..."
python gpt4_correct/models/data_filter.py

# ------------------------------
# Step 4. Correct Reasoning with GPT-4
# ------------------------------
echo "Correcting reasoning chains using GPT-4..."

DATASETS_TO_CORRECT=("logiqa_zh" "Reclor" "LogiQA_v2" "FOLIO")

for NAME in "${DATASETS_TO_CORRECT[@]}"; do
    echo "➡ Correcting $NAME ..."
    python gpt4_correct/models/logic_correction_v2.py \
        --data_path gpt4_correct/Wrong_Inference \
        --dataset_name "$NAME" \
        --api_key "$API_KEY"
done

# ------------------------------
# Step 5. Convert Data to Fine-tuning Format
# ------------------------------
echo "Formatting data for fine-tuning..."
cd data/LOCD/Fintue_data_format
python locd_format.py

echo "All steps completed successfully!"
