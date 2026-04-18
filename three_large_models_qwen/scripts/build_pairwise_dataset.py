#!/usr/bin/env python3
"""Build a pairwise preference dataset from annotation disagreements.

This script converts records from the stage-1 annotation pipeline into a
JSONL dataset with `prompt`, `chosen`, and `rejected` fields that can be used
for DPO or other pairwise-preference training.

Input: annotated_reviews.csv with columns:
    id, review_id, review_text, sentiment (-1/0/1), difficulty, ambiguous_flag,
    verified_by, annotator_a (JSON str), annotator_b (JSON str), judge_reason,
    annotated_at, source

Logic:
    1. Keep only records where annotator_a and annotator_b disagree on sentiment.
    2. The annotator whose sentiment matches the final label becomes chosen;
       the other becomes rejected.
    3. If neither matches the final label, discard the record.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SENTIMENT_MAP = {-1: "negative", 0: "neutral", 1: "positive"}

DEFAULT_INSTRUCTION = (
    "You are given a movie review. Return only valid JSON with the fields "
    "`sentiment`, `difficulty`, `ambiguous_flag`, and "
    "`reason`. `sentiment` must be `positive`, `negative`, or `neutral`. "
    "`difficulty` must be 1, 2, or 3."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pairwise JSONL dataset from annotation disagreements."
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent.parent),
        help="Directory containing annotated_reviews*.csv files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path(__file__).resolve().parent.parent / "data" / "dpo_pairs.jsonl"),
        help="Path to output pairwise JSONL.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction prepended to each prompt.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in ("annotator_a", "annotator_b"):
                raw = row.get(field, "").strip()
                if raw:
                    try:
                        row[field] = json.loads(raw)
                    except json.JSONDecodeError:
                        row[field] = None
                else:
                    row[field] = None

            try:
                row["sentiment"] = int(row["sentiment"])
            except (ValueError, KeyError):
                row["sentiment"] = None
            try:
                row["difficulty"] = int(row["difficulty"])
            except (ValueError, KeyError):
                row["difficulty"] = None

            ambiguous = row.get("ambiguous_flag", "").strip()
            row["ambiguous_flag"] = ambiguous.lower() == "true" if ambiguous else None

            records.append(row)
    return records


def normalize_candidate(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw is None or not isinstance(raw, dict):
        return None

    sentiment = raw.get("sentiment")
    if isinstance(sentiment, (int, float)):
        sentiment = SENTIMENT_MAP.get(int(sentiment))
    elif isinstance(sentiment, str):
        sentiment = sentiment.strip().lower()

    difficulty = raw.get("difficulty")
    if isinstance(difficulty, (int, float)):
        difficulty = int(difficulty)
    else:
        return None

    ambiguous_flag = raw.get("ambiguous_flag")
    if isinstance(ambiguous_flag, str):
        ambiguous_flag = ambiguous_flag.lower() == "true"

    reason = raw.get("reason", "")

    if sentiment not in {"positive", "negative", "neutral"}:
        return None
    if difficulty not in {1, 2, 3}:
        return None

    return {
        "sentiment": sentiment,
        "difficulty": difficulty,
        "ambiguous_flag": ambiguous_flag,
        "reason": reason,
    }


def serialize_response(candidate: dict[str, Any]) -> str:
    payload = {
        "sentiment": candidate["sentiment"],
        "difficulty": candidate["difficulty"],
        "ambiguous_flag": candidate["ambiguous_flag"],
        "reason": candidate["reason"],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    csv_files = sorted(input_dir.glob("annotated_reviews*.csv"))
    if not csv_files:
        print(f"No annotated_reviews*.csv files found in {input_dir}")
        return
    print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

    rows = []
    for csv_file in csv_files:
        rows.extend(read_csv(csv_file))

    kept_pairs: list[dict[str, Any]] = []
    stats = {
        "total_records": 0,
        "kept_pairs": 0,
        "skipped_missing_final": 0,
        "skipped_missing_candidate": 0,
        "skipped_missing_prompt": 0,
        "skipped_no_disagreement": 0,
        "skipped_both_wrong": 0,
    }

    for record in rows:
        stats["total_records"] += 1

        # Final adjudicated sentiment
        final_sentiment = SENTIMENT_MAP.get(record.get("sentiment"))
        if final_sentiment is None:
            stats["skipped_missing_final"] += 1
            continue

        # Normalize annotator outputs
        annotator_a = normalize_candidate(record.get("annotator_a"))
        annotator_b = normalize_candidate(record.get("annotator_b"))
        if annotator_a is None or annotator_b is None:
            stats["skipped_missing_candidate"] += 1
            continue

        # Only keep sentiment disagreements
        if annotator_a["sentiment"] == annotator_b["sentiment"]:
            stats["skipped_no_disagreement"] += 1
            continue

        # Determine chosen/rejected by sentiment match
        a_match = annotator_a["sentiment"] == final_sentiment
        b_match = annotator_b["sentiment"] == final_sentiment

        if a_match:
            chosen, rejected = annotator_a, annotator_b
            chosen_name, rejected_name = "annotator_a", "annotator_b"
        elif b_match:
            chosen, rejected = annotator_b, annotator_a
            chosen_name, rejected_name = "annotator_b", "annotator_a"
        else:
            # Neither matches final label — discard
            stats["skipped_both_wrong"] += 1
            continue

        # Build prompt
        review_text = str(record.get("review_text", "")).strip()
        if not review_text:
            stats["skipped_missing_prompt"] += 1
            continue

        prompt = f"{args.instruction}\n\nReview:\n{review_text}"
        source_id = record.get("id", "")

        pair = {
            "id": f"{source_id}:sentiment_conflict:{chosen_name}_over_{rejected_name}",
            "source_id": source_id,
            "verified_by": record.get("verified_by"),
            "prompt": prompt,
            "chosen": serialize_response(chosen),
            "rejected": serialize_response(rejected),
            "chosen_from": chosen_name,
            "rejected_from": rejected_name,
            "final_sentiment": final_sentiment,
            "review_text": review_text,
        }

        kept_pairs.append(pair)
        stats["kept_pairs"] += 1

    write_jsonl(output_path, kept_pairs)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(kept_pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()
