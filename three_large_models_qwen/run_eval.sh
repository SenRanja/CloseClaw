#!/bin/bash
# ============================================================
# Per-source evaluation on val set
# 填写下方配置后直接 bash run_eval.sh 即可
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============ 在这里修改配置 ============
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"       # base model 路径或 HF ID
ADAPTER=""                                       # LoRA 权重路径，留空则用 base model
BACKEND="vllm"                                   # vllm 或 huggingface
SPLIT="val"                                      # val / test / both
# ========================================

CMD="python scripts/evaluate_per_source.py --split $SPLIT --backend $BACKEND --base-model $BASE_MODEL"

if [ -n "$ADAPTER" ]; then
    CMD="$CMD --adapter $ADAPTER"
fi

echo ">>> Base model: $BASE_MODEL"
echo ">>> Adapter:    ${ADAPTER:-none}"
echo ">>> Backend:    $BACKEND"
echo ""

$CMD
