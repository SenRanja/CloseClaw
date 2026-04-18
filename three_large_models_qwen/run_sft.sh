#!/bin/bash
# ============================================================
# Qwen3 SFT Fine-tuning with LLaMA-Factory
# 填写下方配置后直接 bash run_sft.sh 即可
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============ Eval config ============
BASE_MODEL="/srv/scratch/z5526880/models/Qwen3-4B"
SFT_ADAPTER="output/qwen3-4B/lora/sft"
EVAL_BACKEND="huggingface"
# =====================================

# ---- Step 1: Prepare data ----
echo ">>> Preparing SFT data..."
python scripts/prepare_sft_data.py

# ---- Step 2: Evaluate base model on val (before training) ----
echo ">>> [Before SFT] Running base model prediction pass on val set..."
python scripts/evaluate_sft_val_predictions.py \
    --backend "$EVAL_BACKEND" \
    --base-model "$BASE_MODEL" \
    --output-format csv \
    --run-name before_sft

# ---- Step 3: Run SFT training ----
echo ">>> Starting SFT training..."
cd configs
llamafactory-cli train qwen_sft.yaml
cd "$PROJECT_DIR"

echo ">>> Training complete!"

# ---- Step 4: Evaluate fine-tuned model on val (after training) ----
echo ">>> [After SFT] Running fine-tuned model prediction pass on val set..."
python scripts/evaluate_sft_val_predictions.py \
    --backend "$EVAL_BACKEND" \
    --base-model "$BASE_MODEL" \
    --adapter "$SFT_ADAPTER" \
    --output-format csv \
    --run-name after_sft

# ---- (Optional) Merge LoRA weights ----
# cd configs && llamafactory-cli export qwen_merge.yaml && cd "$PROJECT_DIR"

# ---- (Optional) Chat with the fine-tuned model ----
# cd configs && llamafactory-cli chat qwen_infer.yaml && cd "$PROJECT_DIR"
