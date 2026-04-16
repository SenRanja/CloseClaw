"""
GRPO (Group Relative Policy Optimization) training for sentiment classification.

Loads the SFT model, generates multiple responses per prompt, scores them
with a reward function, and updates the model to prefer correct responses.

Usage:
    python scripts/train_grpo.py
    python scripts/train_grpo.py --base-model /path/to/model --adapter /path/to/sft_lora
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


PROJECT_DIR = Path(__file__).resolve().parent.parent
LABELS = ["positive", "negative", "neutral"]
BOXED_RE = re.compile(r"\\boxed\{(\w+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for sentiment classification.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model path or HF ID.",
    )
    parser.add_argument(
        "--adapter",
        default=str(PROJECT_DIR / "output" / "qwen3-4b" / "lora" / "sft"),
        help="SFT LoRA adapter path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "output" / "qwen3-4b" / "lora" / "grpo"),
        help="Output directory for GRPO LoRA.",
    )
    parser.add_argument(
        "--merged-model-dir",
        default=str(PROJECT_DIR / "output" / "qwen3-4b" / "sft_merged"),
        help="Directory to save merged SFT model (used as GRPO base for evaluation).",
    )
    parser.add_argument("--num-generations", type=int, default=4, help="Number of responses per prompt.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--report-to", default="wandb", help="Reporting backend (wandb/none).")
    return parser.parse_args()


def extract_label(text: str) -> str | None:
    """Extract label from \\boxed{}."""
    match = BOXED_RE.search(text)
    if match:
        candidate = match.group(1).lower()
        if candidate in LABELS:
            return candidate
    return None


SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given movie review into one of three categories:\n"
    "- positive: the reviewer expresses a favorable opinion of the movie.\n"
    "- negative: the reviewer expresses an unfavorable opinion of the movie.\n"
    "- neutral: the reviewer expresses a mixed or balanced opinion with no clear positive or negative leaning.\n"
    "First explain your reasoning, then put your final answer in \\boxed{}, for example \\boxed{positive}."
)


def load_dataset_from_dpo_pairs(data_path: Path, tokenizer) -> Dataset:
    """Load DPO pairs (disagreement samples) and convert to GRPO format.

    Uses only the hard cases where annotators disagreed — these are the samples
    where GRPO can help the most.

    Returns dataset with 'prompt' and 'gold_label' columns.
    """
    prompts = []
    gold_labels = []

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            review_text = pair.get("review_text", "").strip()
            gold = pair.get("final_sentiment", "").strip()
            if not review_text or gold not in LABELS:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify the sentiment of this movie review:\n\n{review_text}"},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            prompts.append(prompt_text)
            gold_labels.append(gold)

    dataset = Dataset.from_dict({"prompt": prompts, "gold_label": gold_labels})
    return dataset


def reward_fn(completions: list[str], gold_label: list[str], **kwargs) -> list[float]:
    """Score completions based on label correctness and reasoning presence.

    Args:
        completions: flat list of generated texts (batch_size * num_generations).
        gold_label: flat list of gold labels (repeated per num_generations by TRL).
    """
    rewards = []
    for completion, gold in zip(completions, gold_label):
        pred = extract_label(completion)

        # Label correctness (main signal)
        label_score = 1.0 if pred == gold else 0.0

        # Reasoning presence (secondary signal)
        boxed_pos = completion.find("\\boxed")
        has_reason = 1.0 if boxed_pos > 20 else 0.0

        # Combined: label 70%, reason 30%
        score = 0.7 * label_score + 0.3 * has_reason
        rewards.append(score)
    return rewards


def main():
    args = parse_args()

    print("=" * 50)
    print("GRPO Training Config")
    print("=" * 50)
    print(f"  Base model:        {args.base_model}")
    print(f"  SFT adapter:       {args.adapter}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Merged model dir:  {args.merged_model_dir}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Grad accum:        {args.gradient_accumulation_steps}")
    print(f"  Effective batch:   {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Num generations:   {args.num_generations}")
    print(f"  Learning rate:     {args.learning_rate}")
    print(f"  Epochs:            {args.num_train_epochs}")
    print(f"  Max new tokens:    {args.max_new_tokens}")
    print(f"  LoRA rank:         {args.lora_rank}")
    print("=" * 50)
    print()

    # Load merged SFT model (must be pre-merged via llamafactory-cli export)
    merged_dir = Path(args.merged_model_dir)
    if not merged_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_dir}. "
            f"Run 'cd configs && llamafactory-cli export qwen_merge.yaml' first."
        )

    # Load tokenizer from merged dir to ensure consistency with model
    print(f"Loading model and tokenizer from {merged_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Check vocab size
    if model.config.vocab_size != len(tokenizer):
        print(f"NOTE: vocab_size in config ({model.config.vocab_size}) != tokenizer ({len(tokenizer)})")
        print("This is normal for Qwen models (config may include padding). Not resizing.")

    # Load dataset (disagreement samples only)
    train_data = load_dataset_from_dpo_pairs(
        PROJECT_DIR / "data" / "dpo_pairs.jsonl", tokenizer
    )
    n_samples = len(train_data)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    estimated_steps = (n_samples // effective_batch) * args.num_train_epochs
    print(f"Training samples:  {n_samples}")
    print(f"Estimated steps:   {estimated_steps} ({n_samples} / {effective_batch} x {args.num_train_epochs} epochs)")
    print()

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.05,
        bf16=False,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=200,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        report_to=args.report_to,
        run_name="qwen3_grpo_sentiment",
    )

    # New LoRA for GRPO on top of merged SFT model
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Workaround: TRL GRPOTrainer expects warnings_issued on model,
    # but PEFT wrapping loses it. Pre-set it to avoid AttributeError.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model()
    print(f"GRPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
