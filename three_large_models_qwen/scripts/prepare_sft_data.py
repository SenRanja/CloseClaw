"""
Convert annotated CSV files to LLaMA-Factory SFT format (Alpaca style).

- Reads all annotated_reviews*.csv files in the project directory.
- SFT data excludes DPO disagreement samples (where annotators disagree on sentiment).
- Val: stratified sample of 500 per source (preserving original label distribution).
- Test: stratified sample of 500 per source. Train: the rest.
- Train and test are merged; val is saved per source for per-source accuracy evaluation.

Output structure:
    data/sft/
      sft_train.json              # all sources merged
      sft_test.json               # all sources merged
      val/
        source_0/sft_val.json     # per source
        source_1/sft_val.json
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SENTIMENT_MAP = {
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
}

SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given movie review into one of three categories:\n"
    "- positive: the reviewer expresses a favorable opinion of the movie.\n"
    "- negative: the reviewer expresses an unfavorable opinion of the movie.\n"
    "- neutral: the reviewer expresses a mixed or balanced opinion with no clear positive or negative leaning.\n"
    "First explain your reasoning, then put your final answer in \\boxed{}, for example \\boxed{positive}."
)


def is_sentiment_disagreement(row: dict) -> bool:
    try:
        a = json.loads(row.get("annotator_a", ""))
        b = json.loads(row.get("annotator_b", ""))
        return a["sentiment"] != b["sentiment"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def get_reason(row: dict) -> str:
    """Get reason from the annotator that matches final sentiment."""
    sentiment = row.get("sentiment", "").strip()
    for field in ("annotator_a", "annotator_b"):
        try:
            ann = json.loads(row.get(field, ""))
            if str(ann.get("sentiment")) == sentiment:
                return ann.get("reason", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return ""


def build_sample(review_text: str, sentiment_label: str, reason: str) -> dict:
    return {
        "instruction": f"Classify the sentiment of this movie review:\n\n{review_text}",
        "input": "",
        "output": f"{reason}\n\n\\boxed{{{sentiment_label}}}",
        "system": SYSTEM_PROMPT,
    }


def stratified_sample(samples: list, n: int, seed: int = 42) -> tuple[list, list]:
    """Sample n items preserving label distribution. Returns (sampled, remaining)."""
    rng = random.Random(seed)

    # Group by label
    by_label = defaultdict(list)
    for s in samples:
        # Extract label from output
        output = s["output"]
        for label in SENTIMENT_MAP.values():
            if f"\\boxed{{{label}}}" in output:
                by_label[label].append(s)
                break

    # Calculate per-label count proportional to distribution
    total = sum(len(v) for v in by_label.values())
    sampled = []
    remaining = []

    for label, items in by_label.items():
        rng.shuffle(items)
        label_n = max(1, round(n * len(items) / total))
        label_n = min(label_n, len(items))
        sampled.extend(items[:label_n])
        remaining.extend(items[label_n:])

    # Adjust if we sampled too many or too few
    rng.shuffle(sampled)
    rng.shuffle(remaining)
    if len(sampled) > n:
        remaining.extend(sampled[n:])
        sampled = sampled[:n]
    elif len(sampled) < n and remaining:
        extra = min(n - len(sampled), len(remaining))
        sampled.extend(remaining[:extra])
        remaining = remaining[extra:]

    return sampled, remaining


def balance_pos_neg(samples: list, seed: int = 44) -> list:
    """Downsample positive to match negative count. Neutral untouched."""
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for s in samples:
        for label in SENTIMENT_MAP.values():
            if f"\\boxed{{{label}}}" in s["output"]:
                by_label[label].append(s)
                break

    neg_count = len(by_label["negative"])

    print(f"\n  Balancing pos/neg in train (target: {neg_count}):")
    print(f"    Before: { {k: len(v) for k, v in sorted(by_label.items())} }")

    rng.shuffle(by_label["positive"])
    balanced = by_label["positive"][:neg_count] + by_label["negative"] + by_label["neutral"]
    rng.shuffle(balanced)

    after = defaultdict(int)
    for s in balanced:
        for label in SENTIMENT_MAP.values():
            if f"\\boxed{{{label}}}" in s["output"]:
                after[label] += 1
                break
    print(f"    After:  { dict(sorted(after.items())) }")

    return balanced


def save_json(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_label_dist(name: str, samples: list):
    """Print label distribution of a sample set."""
    counts = defaultdict(int)
    for s in samples:
        for label in SENTIMENT_MAP.values():
            if f"\\boxed{{{label}}}" in s["output"]:
                counts[label] += 1
                break
    total = sum(counts.values())
    dist = ", ".join(f"{k}={v}({v/total*100:.1f}%)" for k, v in sorted(counts.items()))
    print(f"    {name}: {total} [{dist}]")


def main():
    project_dir = Path(__file__).resolve().parent.parent
    out_dir = project_dir / "data" / "sft"

    # Find all annotated CSV files
    csv_files = sorted(project_dir.glob("annotated_reviews*.csv"))
    if not csv_files:
        print("No annotated_reviews*.csv files found!")
        return

    print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

    # Read all CSVs, group by source
    source_samples: dict[str, list] = defaultdict(list)
    total = 0
    skipped_dpo = 0

    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                sentiment = row.get("sentiment", "").strip()
                review = row.get("review_text", "").strip()
                source = row.get("source", "unknown").strip() or "unknown"

                if sentiment not in SENTIMENT_MAP or not review:
                    continue

                if is_sentiment_disagreement(row):
                    skipped_dpo += 1
                    continue

                reason = get_reason(row)
                sample = build_sample(review, SENTIMENT_MAP[sentiment], reason)
                source_samples[f"source_{source}"].append(sample)

    # Split each source: stratified val 500 → stratified test 500 → train rest
    all_train, all_test, all_val = [], [], []

    for source_name, samples in sorted(source_samples.items()):
        print(f"\n  {source_name} ({len(samples)} samples):")

        # Val: stratified 500
        val, rest = stratified_sample(samples, 500, seed=42)
        # Test: stratified 500 from remaining
        test, train = stratified_sample(rest, 500, seed=43)

        # Save val per source
        save_json(val, out_dir / "val" / source_name / "sft_val.json")

        all_train.extend(train)
        all_test.extend(test)
        all_val.extend(val)

        print_label_dist("val", val)
        print_label_dist("test", test)
        print_label_dist("train", train)

    # Balance positive and negative in train (downsample positive to match negative)
    all_train = balance_pos_neg(all_train, seed=44)

    # Save merged files
    save_json(all_train, out_dir / "sft_train.json")
    save_json(all_test, out_dir / "sft_test.json")

    sft_total = len(all_train) + len(all_test) + len(all_val)
    print(f"\nTotal CSV rows: {total}")
    print(f"Excluded (DPO disagreements): {skipped_dpo}")
    print(f"SFT samples: {sft_total}")
    print(f"  Train: {len(all_train)}, Test: {len(all_test)}, Val: {len(all_val)}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
