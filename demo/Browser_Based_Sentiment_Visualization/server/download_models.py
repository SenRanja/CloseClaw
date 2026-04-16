"""
Pre-download all base models to E:/hf_cache.
Run once before starting the server.
"""

import os
os.environ["HF_HOME"] = "E:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/hf_cache/hub"

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]

for model_id in MODELS:
    print(f"\n{'='*50}")
    print(f"Downloading {model_id} ...")
    print(f"{'='*50}")
    snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.pt", "*.gguf", "original/*"],  # 跳过非必要大文件
    )
    print(f"Done: {model_id}")

print("\nAll models downloaded.")
