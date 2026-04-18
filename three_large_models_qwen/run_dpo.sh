#!/bin/bash
# ============================================================
# Qwen3 DPO Training with LLaMA-Factory
# 需要先完成 SFT 训练（bash run_sft.sh）
# 填写下方配置后直接 bash run_dpo.sh 即可
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============ 在这里修改配置 ============
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"           # base model 路径或 HF ID
SFT_ADAPTER="output/qwen3-4b/lora/sft"              # SFT LoRA 权重路径
DPO_OUTPUT="output/qwen3-4b/lora/dpo"                # DPO 输出路径
# ========================================

echo ">>> Base model:   $BASE_MODEL"
echo ">>> SFT adapter:  $SFT_ADAPTER"
echo ">>> DPO output:   $DPO_OUTPUT"
echo ""

# ---- Step 1: Prepare DPO data ----
echo ">>> Preparing DPO data..."
python scripts/build_pairwise_dataset.py
python scripts/prepare_dpo_data.py

# ---- Step 2: Evaluate SFT model on val (before DPO) ----
echo ">>> [Before DPO] Evaluating SFT model on val set..."
python scripts/evaluate_per_source.py --split val --backend huggingface \
    --base-model "$BASE_MODEL" --adapter "$SFT_ADAPTER"

# ---- Step 3: Generate temp DPO config and train ----
# Convert to absolute paths for yaml
ABS_SFT_ADAPTER="$(cd "$PROJECT_DIR" && realpath "$SFT_ADAPTER")"
ABS_DPO_OUTPUT="$(cd "$PROJECT_DIR" && realpath --canonicalize-missing "$DPO_OUTPUT")"

sed -e "s|^model_name_or_path:.*|model_name_or_path: $BASE_MODEL|" \
    -e "s|^adapter_name_or_path:.*|adapter_name_or_path: $ABS_SFT_ADAPTER|" \
    -e "s|^output_dir:.*|output_dir: $ABS_DPO_OUTPUT|" \
    configs/qwen_dpo.yaml > configs/_qwen_dpo_run.yaml

echo ">>> Starting DPO training..."
cd configs
llamafactory-cli train _qwen_dpo_run.yaml
cd "$PROJECT_DIR"

echo ">>> DPO training complete!"

# ---- Step 4: Evaluate DPO model on val (after DPO) ----
echo ">>> [After DPO] Evaluating DPO model on val set..."
python scripts/evaluate_per_source.py --split val --backend huggingface \
    --base-model "$BASE_MODEL" --adapter "$DPO_OUTPUT"

# ---- (Optional) Merge LoRA weights ----
# cd configs && llamafactory-cli export qwen_merge.yaml && cd "$PROJECT_DIR"
