from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import warnings
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given movie review into one of three categories:\n"
    "- positive: the reviewer expresses a favorable opinion of the movie.\n"
    "- negative: the reviewer expresses an unfavorable opinion of the movie.\n"
    "- neutral: the reviewer expresses a mixed or balanced opinion with no clear positive or negative leaning.\n"
    "First explain your reasoning, then put your final answer in \\boxed{}, for example \\boxed{positive}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive HuggingFace + PEFT LoRA chat.")
    parser.add_argument("--base-model", required=True, help="Base model path or Hugging Face model id.")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--review", help="Run one review and exit instead of interactive mode.")
    parser.add_argument("--raw", action="store_true", help="Treat input as the full user prompt instead of wrapping it.")
    parser.add_argument("--no-system", action="store_true", help="Do not include the sentiment system prompt.")
    parser.add_argument("--verbose", action="store_true", help="Show model loading logs and warnings.")
    return parser.parse_args()


def configure_quiet_mode(verbose: bool) -> None:
    if verbose:
        return

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def resolve_dtype(name: str, torch_module):
    if name == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[name]


def validate_safetensors_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {path}")

    actual_size = path.stat().st_size
    if actual_size < 8:
        raise RuntimeError(f"Adapter weights file is too small or empty: {path} ({actual_size} bytes)")

    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid safetensors header in {path}: {exc}") from exc

    max_data_end = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        data_offsets = metadata.get("data_offsets")
        if not data_offsets or len(data_offsets) != 2:
            raise RuntimeError(f"Invalid data_offsets for tensor {name!r} in {path}")
        max_data_end = max(max_data_end, int(data_offsets[1]))

    expected_size = 8 + header_len + max_data_end
    if actual_size != expected_size:
        raise RuntimeError(
            "Adapter weights file is incomplete or corrupted:\n"
            f"  path: {path}\n"
            f"  actual bytes: {actual_size}\n"
            f"  expected bytes from safetensors header: {expected_size}\n"
            "Re-copy or re-export the full adapter_model.safetensors file from the training machine."
        )


def validate_adapter(adapter: Path) -> None:
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if not config.exists():
        raise FileNotFoundError(f"Adapter config not found: {config}")
    validate_safetensors_file(weights)


def build_user_prompt(text: str, raw: bool) -> str:
    if raw:
        return text
    return f"Classify the sentiment of this movie review:\n\n{text}"


def apply_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, args: argparse.Namespace, text: str) -> str:
    import torch

    messages = []
    if not args.no_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": build_user_prompt(text, args.raw)})

    prompt = apply_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = args.temperature > 0

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    configure_quiet_mode(args.verbose)
    adapter = Path(args.adapter)
    if not adapter.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
    validate_adapter(adapter)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.verbose:
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_dtype(args.torch_dtype, torch),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    if args.review:
        print(generate(model, tokenizer, args, args.review))
        return

    print("Interactive sentiment chat. Paste one review per prompt. Type /exit to quit.")
    while True:
        try:
            text = input("\nreview> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text in {"/exit", "/quit", "exit", "quit"}:
            break
        print(generate(model, tokenizer, args, text))


if __name__ == "__main__":
    main()
