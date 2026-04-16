# Pairwise Dataset Construction for DPO

This project can reuse disagreement cases from the stage-1 annotation pipeline
to build a pairwise preference dataset for DPO.

## Recommended scope

Use disagreement samples where the two annotators do **not** agree. By default,
the conversion script keeps only:

- `sentiment_conflict`: annotator A and annotator B predict different sentiment
  labels.

This is the cleanest source of preference signal because the final adjudicated
label directly tells us which answer is more trustworthy. The script can also
include:

- `difficulty_conflict`: both annotators agree on sentiment but disagree on
  difficulty.

That mode is optional because it teaches preference over the full structured
JSON output, not only over sentiment.

## Pair construction rule

For each disagreement record:

1. Build one candidate response from `annotator_a`.
2. Build one candidate response from `annotator_b`.
3. Compare both candidates against the final adjudicated label
   (`sentiment`, `difficulty`, `ambiguous_flag`).
4. Mark the more aligned candidate as `chosen`.
5. Mark the less aligned candidate as `rejected`.

The current scoring weights are:

- `sentiment_match = 10`
- `difficulty_match = 3`
- `ambiguity_match = 1`

This keeps sentiment as the dominant preference signal while still letting
difficulty and ambiguity contribute when useful.

Although the upstream annotation logs may contain a `confidence` field, it is
treated only as audit metadata. It should not be copied into the downstream
`chosen` / `rejected` target strings for DPO.

## Output schema

Each output row is a JSON object with DPO-friendly fields:

- `prompt`
- `chosen`
- `rejected`

It also keeps traceability fields such as:

- `source_id`
- `pair_type`
- `verified_by`
- `final_label`
- `metadata`

## Example command

```bash
python3 /Users/bowy/Desktop/6713group/scripts/build_pairwise_dataset.py \
  /path/to/annotated_reviews.jsonl \
  /path/to/pairwise_dpo.jsonl
```

To also include difficulty-only disagreements:

```bash
python3 /Users/bowy/Desktop/6713group/scripts/build_pairwise_dataset.py \
  /path/to/annotated_reviews.jsonl \
  /path/to/pairwise_dpo.jsonl \
  --include-difficulty-only
```

## Recommended training usage

For the first DPO run, prefer:

- only `sentiment_conflict` rows
- only records with clear final adjudication
- the raw review text as the prompt context
- the structured JSON answer without `confidence` as the target response

That gives the Qwen model a clean instruction-following objective:
given a review, prefer the answer pattern that better matches the adjudicated
label and reasoning style.
