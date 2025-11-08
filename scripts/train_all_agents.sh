#!/bin/bash
set -e

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

echo "Starting multi-agent training pipeline..."
echo "==========================================="

# Train each agent sequentially
for FEATURE in stylistic syntactic semantic repetition
do
    echo ""
    echo "Training $FEATURE agent..."
    python -m src.training.train_lora \
        --base_model "$BASE_MODEL" \
        --feature "$FEATURE" \
        --output_dir "models/agent_${FEATURE}" \
        --epochs 2 \
        --batch_size 1 \
        --grad_accum 16 \
        --lr 1e-4 \
        --max_length 1536 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05
    
    echo "✓ $FEATURE agent complete!"
done

echo ""
echo "==========================================="
echo "All agents trained successfully!"