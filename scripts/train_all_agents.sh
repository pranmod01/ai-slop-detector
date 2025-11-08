#!/bin/bash
# scripts/train_all_agents.sh

set -e

BASE_MODEL="Qwen/Qwen3-4B"
OUTPUT_BASE="models"

# Use a random port to avoid conflicts
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "Starting multi-agent training on Trainium..."
echo "Using master port: $MASTER_PORT"
echo "=============================================="

for FEATURE in stylistic # syntactic semantic repetition
do
    echo ""
    echo "Training $FEATURE agent..."
    
    torchrun --nproc_per_node=2 src/training/train_lora.py \
        --model_id "$BASE_MODEL" \
        --tokenizer_id "$BASE_MODEL" \
        --feature "$FEATURE" \
        --output_dir "${OUTPUT_BASE}/agent_${FEATURE}" \
        --max_seq_length 1024 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1e-4 \
        --bf16 \
        --logging_steps 10 \
        --save_strategy epoch \
        --eval_strategy no \
        --do_eval False \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --seed 42 \
        --report_to none | tee "logs/${FEATURE}_train.log"
    
    echo "✓ $FEATURE agent complete!"
done

echo ""
echo "=============================================="
echo "All agents trained successfully!"