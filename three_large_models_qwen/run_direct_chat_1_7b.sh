#!/usr/bin/env bash
# Direct interactive inference without LLaMA-Factory chat.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/srv/scratch/cruise/kejingwang/Qwen31.7b}"
ADAPTER="${ADAPTER:-$PROJECT_DIR/output/qwen3-1.7B/lora/sft}"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$PYTHON_BIN" scripts/chat_hf_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  "$@"
