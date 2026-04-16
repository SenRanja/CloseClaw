"""
Evaluate the fine-tuned model per source on test and val sets.

Reports accuracy, precision, recall, F1 for each source and overall.
Supports both vLLM (fast, default) and HuggingFace backends.

Usage:
    python scripts/evaluate_per_source.py                        # vllm, eval all
    python scripts/evaluate_per_source.py --split val            # vllm, eval val only
    python scripts/evaluate_per_source.py --backend huggingface  # use HF backend
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABELS = ["positive", "negative", "neutral"]
BOXED_RE = re.compile(r"\\boxed\{(\w+)\}")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = str(PROJECT_DIR / "output" / "qwen3-4b" / "lora" / "sft")
DATA_DIR = PROJECT_DIR / "data" / "sft"
RESULTS_DIR = PROJECT_DIR / "output" / "eval_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-source evaluation.")
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL, help="Base model path or HF ID."
    )
    parser.add_argument(
        "--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter path."
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "both"],
        default="both",
        help="Which split(s) to evaluate.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "huggingface"],
        default="vllm",
        help="Inference backend.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max tokens to generate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (HF only, vLLM handles batching internally)."
    )
    return parser.parse_args()


# ===================== vLLM backend =====================

def load_model_vllm(base_model: str, adapter: str):
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    adapter_path = Path(adapter)
    lora_request = None
    if adapter_path.exists():
        print(f"Loading model with LoRA adapter: {adapter}")
        from vllm.lora.request import LoRARequest
        llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
        )
        lora_request = LoRARequest("sft_adapter", 1, adapter)
    else:
        print(f"Adapter not found at {adapter}, using base model only.")
        llm = LLM(model=base_model, trust_remote_code=True)

    return llm, tokenizer, lora_request


def predict_vllm(llm, tokenizer, lora_request, samples: list[dict], max_new_tokens: int) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )

    prompts = []
    for s in samples:
        messages = build_chat_messages(s)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    predictions = []
    for output in outputs:
        pred = output.outputs[0].text
        predictions.append(extract_label(pred))
    return predictions


# ===================== HuggingFace backend =====================

def load_model_hf(base_model: str, adapter: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    if Path(adapter).exists():
        print(f"Loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    else:
        print(f"Adapter not found at {adapter}, using base model only.")
    model.eval()
    return model, tokenizer


def predict_batch_hf(model, tokenizer, samples: list[dict], max_new_tokens: int) -> list[str]:
    all_messages = [build_chat_messages(s) for s in samples]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    predictions = []
    for i, output in enumerate(outputs):
        new_tokens = output[inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(extract_label(pred))
    return predictions


# ===================== Common =====================

def extract_label(text: str) -> str:
    """Extract label from model output. Tries \\boxed{} first, then substring match."""
    text = text.strip().lower()
    match = BOXED_RE.search(text)
    if match:
        candidate = match.group(1)
        if candidate in LABELS:
            return candidate
    for label in LABELS:
        if label in text:
            return label
    return text


def build_chat_messages(sample: dict) -> list[dict]:
    messages = []
    if sample.get("system"):
        messages.append({"role": "system", "content": sample["system"]})
    prompt = sample["instruction"]
    if sample.get("input"):
        prompt += f"\n{sample['input']}"
    messages.append({"role": "user", "content": prompt})
    return messages


def compute_metrics(golds: list[str], preds: list[str]) -> dict:
    correct = sum(g == p for g, p in zip(golds, preds))
    total = len(golds)
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(g == label for g in golds),
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total": total,
        "correct": correct,
        "per_class": per_class,
    }


def find_eval_files(split_name: str) -> dict[str, Path]:
    """Find eval files. Val: per-source + merged. Test: merged only."""
    files = {}

    if split_name == "val":
        val_dir = DATA_DIR / "val"
        if val_dir.exists():
            for source_dir in sorted(val_dir.iterdir()):
                if source_dir.is_dir():
                    f = source_dir / "sft_val.json"
                    if f.exists():
                        files[source_dir.name] = f
    else:
        merged = DATA_DIR / "sft_test.json"
        if merged.exists():
            files["all_sources"] = merged

    return files


def evaluate_split(predict_fn, split_name: str, files: dict[str, Path], args) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {split_name}")
    print(f"{'='*60}")

    results = {}
    for source_name, filepath in files.items():
        with open(filepath, encoding="utf-8") as f:
            samples = json.load(f)

        golds = [extract_label(s["output"]) for s in samples]
        preds = predict_fn(samples)

        metrics = compute_metrics(golds, preds)
        results[source_name] = metrics

        print(f"\n  [{source_name}] accuracy={metrics['accuracy']}, macro_f1={metrics['macro_f1']}, n={metrics['total']}")
        for label, m in metrics["per_class"].items():
            print(f"    {label:>10s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (n={m['support']})")

    return results


def main():
    args = parse_args()

    if args.backend == "vllm":
        llm, tokenizer, lora_request = load_model_vllm(args.base_model, args.adapter)

        def predict_fn(samples):
            return predict_vllm(llm, tokenizer, lora_request, samples, args.max_new_tokens)
    else:
        model, tokenizer = load_model_hf(args.base_model, args.adapter)

        def predict_fn(samples):
            preds = []
            for i in range(0, len(samples), args.batch_size):
                batch = samples[i : i + args.batch_size]
                preds.extend(predict_batch_hf(model, tokenizer, batch, args.max_new_tokens))
                if (i // args.batch_size) % 10 == 0:
                    print(f"    progress: {min(i + args.batch_size, len(samples))}/{len(samples)}")
            return preds

    splits = ["test", "val"] if args.split == "both" else [args.split]
    all_results = {}

    for split_name in splits:
        files = find_eval_files(split_name)
        if not files:
            print(f"No data files found for {split_name}, skipping.")
            continue

        all_results[split_name] = evaluate_split(predict_fn, split_name, files, args)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "per_source_metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
