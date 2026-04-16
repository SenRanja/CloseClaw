#!/bin/bash
# ============================================================
# Qwen3 GRPO Training
# 需要先完成 SFT 训练（bash run_sft.sh）
# 填写下方配置后直接 bash run_grpo.sh 即可
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============ 在这里修改配置 ============
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER="output/qwen3-4b/lora/sft"
GRPO_OUTPUT="output/qwen3-4b/lora/grpo"
MERGED_MODEL="output/qwen3-4b/sft_merged"
BACKEND="huggingface"

# GRPO 训练参数
NUM_GENERATIONS=4                # 每个 prompt 生成几个回答
BATCH_SIZE=16                    # 每步处理几个 prompt
GRAD_ACCUM=1                     # 梯度累积步数，有效 batch = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE=5e-6               # 学习率
EPOCHS=1                         # 训练轮数
MAX_NEW_TOKENS=256               # 最大生成长度
LORA_RANK=8                      # LoRA rank
REPORT_TO="wandb"                # wandb 或 none
# ========================================

echo ">>> Base model:       $BASE_MODEL"
echo ">>> SFT adapter:      $SFT_ADAPTER"
echo ">>> GRPO output:      $GRPO_OUTPUT"
echo ">>> Batch size:       $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo ">>> Num generations:  $NUM_GENERATIONS"
echo ">>> Learning rate:    $LEARNING_RATE"
echo ">>> Epochs:           $EPOCHS"
echo ""

# ---- Step 1: Prepare data (if needed) ----
if [ ! -f data/dpo_pairs.jsonl ]; then
    echo ">>> Generating DPO pairs data..."
    python scripts/build_pairwise_dataset.py
fi

# ---- Step 2: Merge SFT model (if needed) ----
if [ ! -d "$MERGED_MODEL" ]; then
    echo ">>> Merging SFT adapter into base model..."
    sed -e "s|^model_name_or_path:.*|model_name_or_path: $BASE_MODEL|" \
        -e "s|^adapter_name_or_path:.*|adapter_name_or_path: $(realpath "$SFT_ADAPTER")|" \
        -e "s|^export_dir:.*|export_dir: $(realpath --canonicalize-missing "$MERGED_MODEL")|" \
        configs/qwen_merge.yaml > configs/_qwen_merge_run.yaml
    cd configs
    llamafactory-cli export _qwen_merge_run.yaml
    cd "$PROJECT_DIR"
fi

# ---- Step 3: Run GRPO training ----
echo ">>> Starting GRPO training..."
python scripts/train_grpo.py \
    --base-model "$BASE_MODEL" \
    --adapter "$SFT_ADAPTER" \
    --output-dir "$GRPO_OUTPUT" \
    --merged-model-dir "$MERGED_MODEL" \
    --num-generations "$NUM_GENERATIONS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --num-train-epochs "$EPOCHS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --lora-rank "$LORA_RANK" \
    --report-to "$REPORT_TO"

echo ">>> GRPO training complete!"

# ---- Step 4: Evaluate GRPO model on val (after GRPO) ----
echo ">>> [After GRPO] Evaluating GRPO model on val set..."
python scripts/evaluate_per_source.py --split val --backend "$BACKEND" \
    --base-model "$MERGED_MODEL" --adapter "$GRPO_OUTPUT"
