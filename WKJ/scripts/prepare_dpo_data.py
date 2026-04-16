"""
Convert dpo_pairs.jsonl to LLaMA-Factory DPO format (ShareGPT style).

Balances rejected label distribution to prevent the model from
over-penalizing any single label (especially neutral).

Input:  data/dpo_pairs.jsonl
Output: data/dpo/dpo_train.json
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given movie review into one of three categories:\n"
    "- positive: the reviewer expresses a favorable opinion of the movie.\n"
    "- negative: the reviewer expresses an unfavorable opinion of the movie.\n"
    "- neutral: the reviewer expresses a mixed or balanced opinion with no clear positive or negative leaning.\n"
    "First explain your reasoning, then put your final answer in \\boxed{}, for example \\boxed{positive}."
)

BOXED_RE = re.compile(r"\\boxed\{(\w+)\}")


def extract_rejected_label(sample: dict) -> str:
    match = BOXED_RE.search(sample["rejected"]["value"])
    return match.group(1) if match else "unknown"


def extract_chosen_label(sample: dict) -> str:
    match = BOXED_RE.search(sample["chosen"]["value"])
    return match.group(1) if match else "unknown"


def main():
    project_dir = Path(__file__).resolve().parent.parent
    input_path = project_dir / "data" / "dpo_pairs.jsonl"
    out_dir = project_dir / "data" / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build all samples
    all_samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            chosen_obj = json.loads(pair["chosen"])
            rejected_obj = json.loads(pair["rejected"])

            chosen_text = f"{chosen_obj['reason']}\n\n\\boxed{{{chosen_obj['sentiment']}}}"
            rejected_text = f"{rejected_obj['reason']}\n\n\\boxed{{{rejected_obj['sentiment']}}}"

            sample = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"Classify the sentiment of this movie review:\n\n{pair['review_text']}"},
                ],
                "chosen": {"from": "gpt", "value": chosen_text},
                "rejected": {"from": "gpt", "value": rejected_text},
            }
            all_samples.append(sample)

    # Group by rejected label AND chosen label
    rng = random.Random(42)

    by_rejected = defaultdict(list)
    by_chosen = defaultdict(list)
    for s in all_samples:
        rej_label = extract_rejected_label(s)
        cho_label = extract_chosen_label(s)
        by_rejected[rej_label].append(s)
        by_chosen[cho_label].append(s)

    print("Before balancing:")
    print("  Rejected:", {k: len(v) for k, v in sorted(by_rejected.items())})
    print("  Chosen:  ", {k: len(v) for k, v in sorted(by_chosen.items())})

    # Step 1: Cap rejected by median
    rej_counts = sorted(len(v) for v in by_rejected.values())
    rej_cap = rej_counts[len(rej_counts) // 2]

    capped_ids = set()
    for label, samples in by_rejected.items():
        rng.shuffle(samples)
        for s in samples[:rej_cap]:
            capped_ids.add(id(s))

    after_rej_cap = [s for s in all_samples if id(s) in capped_ids]

    # Step 2: Cap chosen by median on the remaining set
    by_chosen_2 = defaultdict(list)
    for s in after_rej_cap:
        cho_label = extract_chosen_label(s)
        by_chosen_2[cho_label].append(s)

    cho_counts = sorted(len(v) for v in by_chosen_2.values())
    cho_cap = cho_counts[len(cho_counts) // 2]

    balanced = []
    for label, samples in by_chosen_2.items():
        rng.shuffle(samples)
        balanced.extend(samples[:cho_cap])

    rng.shuffle(balanced)

    # Print final distribution
    final_rej = defaultdict(int)
    final_cho = defaultdict(int)
    for s in balanced:
        final_rej[extract_rejected_label(s)] += 1
        final_cho[extract_chosen_label(s)] += 1

    print(f"\nAfter balancing (rej_cap={rej_cap}, cho_cap={cho_cap}):")
    print("  Rejected:", dict(sorted(final_rej.items())))
    print("  Chosen:  ", dict(sorted(final_cho.items())))

    with open(out_dir / "dpo_train.json", "w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)

    print(f"\nTotal DPO samples: {len(balanced)}")
    print(f"Saved to {out_dir / 'dpo_train.json'}")


if __name__ == "__main__":
    main()
