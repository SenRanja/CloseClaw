"""
Run one inference pass on the two SFT validation source files and save predictions.

The default output is a CSV file. Each row contains exactly:
    id, review, label, prediction

Examples:
    # Before SFT: base model only
    python scripts/evaluate_sft_val_predictions.py --run-name before_sft

    # After SFT: base model + LoRA adapter
    python scripts/evaluate_sft_val_predictions.py \
        --adapter output/qwen3-4b/lora/sft \
        --run-name after_sft
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable


LABELS = ["positive", "negative", "neutral"]
BOXED_RE = re.compile(r"\\boxed\{(\w+)\}")
LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")
REVIEW_PREFIX = "Classify the sentiment of this movie review:\n\n"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_VAL_DIR = PROJECT_DIR / "data" / "sft" / "val"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "eval_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save per-sample predictions on data/sft/val source files."
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model path or HF ID.",
    )
    parser.add_argument(
        "--adapter",
        default="",
        help="LoRA adapter path for the after-SFT model. Leave empty for before-SFT/base model.",
    )
    parser.add_argument(
        "--backend",
        choices=["huggingface", "vllm"],
        default="huggingface",
        help="Inference backend.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="Directory containing source_*/sft_val.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save prediction files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <output-dir>/<run-name>_val_predictions.<format>.",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default="csv",
        help="Prediction file format.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name used in the default output filename. Defaults to before_sft or after_sft.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the HuggingFace backend.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit over the loaded validation samples.",
    )
    return parser.parse_args()


def normalize_adapter(adapter: str) -> str | None:
    adapter = adapter.strip()
    if adapter.lower() in {"", "none", "null"}:
        return None
    return adapter


def extract_label(text: str) -> str:
    text = text.strip().lower()
    match = BOXED_RE.search(text)
    if match:
        candidate = match.group(1)
        if candidate in LABELS:
            return candidate

    label_matches = LABEL_RE.findall(text)
    if label_matches:
        return label_matches[-1]

    return text


def extract_review(sample: dict) -> str:
    instruction = sample.get("instruction", "")
    if instruction.startswith(REVIEW_PREFIX):
        review = instruction[len(REVIEW_PREFIX) :]
    else:
        review = instruction

    extra_input = sample.get("input", "")
    if extra_input:
        review = f"{review}\n{extra_input}"

    return review.strip()


def build_chat_messages(sample: dict) -> list[dict]:
    messages = []
    if sample.get("system"):
        messages.append({"role": "system", "content": sample["system"]})

    prompt = sample["instruction"]
    if sample.get("input"):
        prompt += f"\n{sample['input']}"

    messages.append({"role": "user", "content": prompt})
    return messages


def get_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def load_val_records(val_dir: Path, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    samples: list[dict] = []
    records: list[dict] = []

    for source_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        source_file = source_dir / "sft_val.json"
        if not source_file.exists():
            continue

        with open(source_file, encoding="utf-8") as f:
            source_samples = json.load(f)

        for idx, sample in enumerate(source_samples, start=1):
            samples.append(sample)
            records.append(
                {
                    "id": f"{source_dir.name}_{idx:06d}",
                    "review": extract_review(sample),
                    "label": extract_label(sample["output"]),
                    "prediction": "",
                }
            )

    if not records:
        raise FileNotFoundError(f"No source_*/sft_val.json files found in {val_dir}")

    if limit is not None:
        samples = samples[:limit]
        records = records[:limit]

    return samples, records


def load_predictor_hf(
    base_model: str,
    adapter: str | None,
    max_new_tokens: int,
    batch_size: int,
) -> Callable[[list[dict]], list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    else:
        print("No adapter provided; using base model only.")

    model.eval()
    input_device = get_model_device(model)

    def predict(samples: list[dict]) -> list[str]:
        predictions: list[str] = []

        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            messages = [build_chat_messages(sample) for sample in batch]
            texts = [
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for message in messages
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(input_device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(extract_label(text) for text in decoded)

            done = min(start + batch_size, len(samples))
            print(f"  progress: {done}/{len(samples)}")

        return predictions

    return predict


def load_predictor_vllm(
    base_model: str,
    adapter: str | None,
    max_new_tokens: int,
) -> Callable[[list[dict]], list[str]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    lora_request = None

    if adapter:
        from vllm.lora.request import LoRARequest

        print(f"Loading model with LoRA adapter: {adapter}")
        llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
        )
        lora_request = LoRARequest("sft_adapter", 1, adapter)
    else:
        print(f"Loading base model without adapter: {base_model}")
        llm = LLM(model=base_model, trust_remote_code=True)

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    def predict(samples: list[dict]) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(sample),
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in samples
        ]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        return [extract_label(output.outputs[0].text) for output in outputs]

    return predict


def resolve_output_path(args: argparse.Namespace, adapter: str | None) -> Path:
    if args.output_file is not None:
        if args.output_file.is_absolute():
            return args.output_file
        return PROJECT_DIR / args.output_file

    run_name = args.run_name or ("after_sft" if adapter else "before_sft")
    return args.output_dir / f"{run_name}_val_predictions.{args.output_format}"


def save_predictions(
    records: list[dict],
    predictions: list[str],
    output_path: Path,
    output_format: str,
) -> None:
    if len(records) != len(predictions):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match records ({len(records)})."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for record, prediction in zip(records, predictions):
        record["prediction"] = prediction

    if output_format == "csv":
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "review", "label", "prediction"])
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    adapter = normalize_adapter(args.adapter)

    samples, records = load_val_records(args.val_dir, args.limit)
    print(f"Loaded {len(records)} validation samples from {args.val_dir}")

    if args.backend == "vllm":
        predict = load_predictor_vllm(args.base_model, adapter, args.max_new_tokens)
    else:
        predict = load_predictor_hf(
            args.base_model,
            adapter,
            args.max_new_tokens,
            args.batch_size,
        )

    predictions = predict(samples)
    output_path = resolve_output_path(args, adapter)
    save_predictions(records, predictions, output_path, args.output_format)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
