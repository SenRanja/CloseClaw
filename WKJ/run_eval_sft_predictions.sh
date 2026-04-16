#!/bin/bash
# ============================================================
# Evaluate SFT LoRA adapter on data/sft/val and save predictions.
# 只跑训练后权重评测，不重新准备数据、不重新训练。
#
# Usage:
#   bash run_eval_sft_predictions.sh
#
# Override examples:
#   BASE_MODEL=/srv/scratch/z5526880/models/Qwen3-0.6B \
#   SFT_ADAPTER=output/qwen3-0.6b/lora/sft \
#   bash run_eval_sft_predictions.sh
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============ Eval config ============
BASE_MODEL=/srv/scratch/z5526880/models/Qwen3-0.6B
SFT_ADAPTER=/srv/scratch/z5526880/6713group/output/qwen3-0.6B/lora/sft
BACKEND=huggingface            # huggingface / vllm
VAL_DIR=/srv/scratch/z5526880/6713group/data/sft/val
OUTPUT_DIR=/srv/scratch/z5526880/6713group/output/eval_predictions
OUTPUT_FORMAT=csv        # csv / json
RUN_NAME=after_sft
BATCH_SIZE=8
MAX_NEW_TOKENS=256
LIMIT=
# =====================================

echo ">>> Evaluating SFT model on val set"
echo ">>> Base model:  $BASE_MODEL"
echo ">>> Adapter:     $SFT_ADAPTER"
echo ">>> Backend:     $BACKEND"
echo ">>> Val dir:     $VAL_DIR"
echo ">>> Output dir:  $OUTPUT_DIR"
echo ">>> Run name:    $RUN_NAME"
echo ""

if [ ! -d "$SFT_ADAPTER" ]; then
    echo "ERROR: SFT_ADAPTER directory not found: $SFT_ADAPTER"
    echo "       Set SFT_ADAPTER to the LoRA output_dir or a checkpoint directory."
    echo "       You can search with: find output -name adapter_config.json -print"
    exit 1
fi

if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    if ! find "$SFT_ADAPTER" -maxdepth 2 -path "$SFT_ADAPTER/checkpoint-*/adapter_config.json" -print -quit | grep -q .; then
        echo "ERROR: adapter_config.json not found in $SFT_ADAPTER"
        echo "       Also found no checkpoint-*/adapter_config.json under it."
        echo "       Set SFT_ADAPTER to the directory that contains adapter_config.json."
        exit 1
    fi
    echo ">>> adapter_config.json not found at adapter root; evaluation script will use the latest checkpoint-* adapter."
fi

CMD=(
    python scripts/evaluate_sft_val_predictions.py
    --backend "$BACKEND"
    --base-model "$BASE_MODEL"
    --adapter "$SFT_ADAPTER"
    --val-dir "$VAL_DIR"
    --output-dir "$OUTPUT_DIR"
    --output-format "$OUTPUT_FORMAT"
    --run-name "$RUN_NAME"
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
)

if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
