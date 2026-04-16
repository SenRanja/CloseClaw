# 3.3 Annotation Process

## Label Schema

Each movie review is assigned one of three sentiment labels: positive (1), negative (−1), or mixed (0). A review is labeled positive if the dominant overall impression of the film is favorable, negative if unfavorable, and mixed only when the review contains substantial positive and negative judgments about the film itself with no clear dominant side. Crucially, sentiment is assessed with respect to the movie being reviewed, not ancillary targets such as the source novel, DVD extras, or an actor's other works. Two additional metadata fields are recorded per sample: a difficulty score (1 = explicit/easy, 2 = contrastive/mixed context, 3 = implicit/sarcastic) and an ambiguous flag indicating whether a reasonable annotator could plausibly assign more than one label even after target disambiguation.

## Multi-Model Annotation

Annotation is performed by two independent LLM annotators running in parallel: GPT-4o-mini (Annotator A) and Gemini 2.5 Flash (Annotator B). Both models receive the same structured system prompt specifying label definitions, a difficulty rubric, disambiguation rules, and five worked examples. Each annotator returns a structured JSON object containing sentiment, difficulty, ambiguous_flag, and a brief reason.

## Consensus and Dispute Resolution

After both annotators produce their outputs, a three-branch consensus router determines the final label:

- Strong consensus: if both annotators agree on both sentiment and difficulty, their shared result is accepted directly.

- Weak consensus: if sentiment agrees but difficulty scores differ, the sentiment label is accepted and the higher difficulty score is adopted (conservative upgrade).

- Zero consensus (Judge arbitration): if the two annotators disagree on sentiment, a third, more powerful model — GPT-4o acting as Supreme Judge — independently annotates the same review using the identical prompt and returns the final label along with an explicit justification (judge_reason). The verified_by field records which resolution branch was taken (strong_consensus, weak_consensus_upgraded, or supreme_judge), providing full traceability.

## Robustness Mechanisms

Each review is processed with up to three retry attempts with exponential back-off to handle transient API errors. The pipeline supports checkpoint-resume: already-annotated review IDs are tracked in the output CSV, so interrupted runs can resume without duplication. Concurrent annotation is handled via a thread pool (default concurrency = 8), with a mutex-protected CSV writer ensuring safe parallel writes.