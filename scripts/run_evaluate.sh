#!/bin/bash

set -e

# Check if model_path is provided
if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_evaluation.sh <merged_model_path>"
    exit 1
fi

MODEL_PATH=$1

# Define output paths
OUTPUT_PATH=./results/generated_data
RESULT_PATH=./results/accuracy.json

# Create results directory if it doesn't exist
mkdir -p ./results

echo "=============================================="
echo "Starting evaluation..."
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Result path: $RESULT_PATH"
echo "=============================================="

# Run the evaluation script
python logic_llm/evaluate/evaluate.py \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --result_path "$RESULT_PATH"

echo "Evaluation finished! Results saved to $RESULT_PATH"
