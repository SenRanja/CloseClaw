"""
Pre-download all base models to E:/hf_cache.
Run once before starting the server.
"""

import os
os.environ["HF_HOME"] = "E:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/hf_cache/hub"

from huggingface_hub import snapshot_download

BASE_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]

LORA_ADAPTERS = [
    "alanwang2001/qwen3-0.6B-sentiment-lora",
    "alanwang2001/qwen3-1.7B-sentiment-lora",
]

for model_id in BASE_MODELS:
    print(f"\n{'='*50}")
    print(f"Downloading {model_id} ...")
    print(f"{'='*50}")
    snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.pt", "*.gguf", "original/*"],
    )
    print(f"Done: {model_id}")

for adapter_id in LORA_ADAPTERS:
    print(f"\n{'='*50}")
    print(f"Downloading LoRA adapter {adapter_id} ...")
    print(f"{'='*50}")
    snapshot_download(repo_id=adapter_id)
    print(f"Done: {adapter_id}")

print("\nAll models downloaded.")
