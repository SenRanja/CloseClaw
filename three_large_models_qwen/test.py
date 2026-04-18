"""
COMP6713 Sentiment Annotation Pipeline
Three-class version: positive / negative / mixed
"""

import os
import json
import sqlite3
import hashlib
import logging
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("annotation_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

VALID_SENTIMENTS = {"positive", "negative", "mixed"}


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class AnnotationResult:
    sentiment: str   # "positive" | "negative" | "mixed"
    difficulty: int  # 1 | 2 | 3
    ambiguous_flag: bool


@dataclass
class FinalRecord:
    id: str
    review_id: str
    review_text: str
    sentiment: str
    difficulty: int
    ambiguous_flag: bool
    verified_by: str
    annotator_a: dict
    annotator_b: dict
    judge_reason: Optional[str]
    annotated_at: str


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

ANNOTATOR_SYSTEM = """\
You are a sentiment annotation expert for movie reviews.

## Task
Analyze the given movie review and return a JSON object with exactly these fields:
- sentiment: "positive", "negative", or "mixed"
- difficulty: 1, 2, or 3 (see rubric below)
- ambiguous_flag: true or false
- reason: a brief justification

## Core Rule
Assign sentiment specifically to the MOVIE being reviewed.

If the review mentions other targets, such as:
- the book, novel, source material, remake, or adaptation
- packaging, VHS/DVD extras, marketing, or release details
- actors' other works or unrelated side topics

then ignore or downweight sentiment directed at those targets.
Prioritize the reviewer's overall attitude toward the movie itself.

## Sentiment Definitions
- "positive": the dominant overall impression of the film is favorable
- "negative": the dominant overall impression of the film is unfavorable
- "mixed": the review contains substantial positive AND negative judgments about the film itself, with no clear dominant side

Important:
- Do NOT label as "mixed" merely because the review contains both positive and negative words
- Do NOT label as "mixed" if positive sentiment is directed at another target (for example, the book) while the movie itself is clearly criticized
- If the movie is clearly recommended or clearly discouraged overall, prefer "positive" or "negative"

## High-Weight Evidence
Give strong weight to direct recommendation language about the movie, such as:
- "must-see", "highly recommend", "worth watching"
- "avoid it", "steer clear", "skip it", "waste of time", "not worth it"

## Difficulty Rubric
Level 1 (Easy — Explicit):
  Explicit strong sentiment words (e.g., masterpiece, awful). Simple syntax.
  A TF-IDF + Logistic Regression baseline would classify this reliably.

Level 2 (Medium — Mixed Context):
  Contrastive connectives (but, although, however), double negatives,
  substantive pros-and-cons structure, or target disambiguation.
  Note: generic hedges like "not for everyone" or "some may disagree"
  do NOT qualify — there must be real positive AND negative content,
  or real interpretation/disambiguation difficulty.

Level 3 (Hard — Implicit / Sarcastic):
  No explicit sentiment anchors, or sentiment conveyed via irony,
  hyperbole, metaphor, or world knowledge.
  Example: "This movie cured my insomnia."

Important:
Difficulty measures how hard the review is to interpret.
It is NOT the same thing as ambiguity.
A review can be difficulty 2 or 3 and still have a clear sentiment label.

## Ambiguous Flag
Set ambiguous_flag to true only if, after focusing on the movie itself,
a reasonable annotator could still plausibly choose more than one sentiment label.

Set ambiguous_flag to false when the overall movie sentiment is clear,
even if:
- the review is long or messy
- there are minor caveats
- other targets are discussed
- the wording is nuanced or contrastive

Do NOT use ambiguous_flag as a confidence score.

## Reason
The reason must be brief and specific:
- refer to the movie sentiment only
- mention the strongest textual evidence
- mention target disambiguation if relevant
- use at most 2 short sentences

## Output Format
Return ONLY valid JSON with no markdown fences and no extra text.
Do NOT include any fields other than:
- sentiment
- difficulty
- ambiguous_flag
- reason

## Examples
Review: "An absolute masterpiece. Every scene is perfect."
Output: {{"sentiment":"positive","difficulty":1,"ambiguous_flag":false,"reason":"The review gives direct and unambiguous praise for the movie."}}

Review: "The acting is superb, but the plot drags so badly it nearly ruins the film."
Output: {{"sentiment":"negative","difficulty":2,"ambiguous_flag":false,"reason":"The review notes one strength, but the overall judgment is negative because the plot nearly ruins the film."}}

Review: "I can't believe how riveting it was to watch paint dry for two hours."
Output: {{"sentiment":"negative","difficulty":3,"ambiguous_flag":false,"reason":"The review uses sarcasm to convey that the movie was extremely boring."}}

Review: "The cinematography is breathtaking and the score is flawless, but the script is an incoherent mess and the lead actor is completely miscast."
Output: {{"sentiment":"mixed","difficulty":2,"ambiguous_flag":true,"reason":"The movie receives strong praise for style and strong criticism for writing and casting, with no clear dominant side."}}

Review: "The book is excellent, but the movie is a mess. Read the novel and skip the film."
Output: {{"sentiment":"negative","difficulty":2,"ambiguous_flag":false,"reason":"The positive sentiment is directed at the book, while the movie is clearly criticized and explicitly discouraged."}}
"""

HUMAN_TEMPLATE = "Review:\n{review_text}"


# ──────────────────────────────────────────────
# Chain construction
# ──────────────────────────────────────────────

def build_annotator_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANNOTATOR_SYSTEM),
        ("human", HUMAN_TEMPLATE),
    ])
    return prompt | llm | JsonOutputParser()


def build_judge_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANNOTATOR_SYSTEM),
        ("human", HUMAN_TEMPLATE),
    ])
    return prompt | llm | JsonOutputParser()


def build_parallel_annotators(chain_a, chain_b):
    return RunnableParallel(result_a=chain_a, result_b=chain_b)


# ──────────────────────────────────────────────
# Parsing and validation
# ──────────────────────────────────────────────

def parse_annotation(raw: dict) -> AnnotationResult:
    sentiment = str(raw.get("sentiment", "")).lower().strip()
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"Invalid sentiment value: '{sentiment}', must be positive / negative / mixed")
    return AnnotationResult(
        sentiment=sentiment,
        difficulty=int(raw.get("difficulty", 2)),
        ambiguous_flag=bool(raw.get("ambiguous_flag", False)),
    )


# ──────────────────────────────────────────────
# Three-branch consensus routing
# ──────────────────────────────────────────────

async def consensus_router(
    review_text: str,
    raw_a: dict,
    raw_b: dict,
    judge_chain,
) -> tuple[AnnotationResult, str, Optional[str]]:
    a = parse_annotation(raw_a)
    b = parse_annotation(raw_b)

    # Branch 1: strong consensus
    if a.sentiment == b.sentiment and a.difficulty == b.difficulty:
        final = AnnotationResult(
            sentiment=a.sentiment,
            difficulty=a.difficulty,
            ambiguous_flag=a.ambiguous_flag or b.ambiguous_flag,
        )
        return final, "strong_consensus", None

    # Branch 2: weak consensus (same sentiment, different difficulty) → take higher
    if a.sentiment == b.sentiment and a.difficulty != b.difficulty:
        final = AnnotationResult(
            sentiment=a.sentiment,
            difficulty=max(a.difficulty, b.difficulty),
            ambiguous_flag=a.ambiguous_flag or b.ambiguous_flag,
        )
        return final, "weak_consensus_upgraded", None

    # Branch 3: zero consensus (sentiment conflict) → Judge
    log.info("  → Zero consensus: invoking Judge...")
    judge_raw = await judge_chain.ainvoke({
        "review_text": review_text,
    })
    judge_sentiment = str(judge_raw.get("sentiment", "")).lower().strip()
    if judge_sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"Judge returned invalid sentiment: '{judge_sentiment}'")
    judge_reason = judge_raw.get("reason", None)
    final = AnnotationResult(
        sentiment=judge_sentiment,
        difficulty=int(judge_raw.get("difficulty", 2)),
        ambiguous_flag=bool(judge_raw.get("ambiguous_flag", False)),
    )
    return final, "supreme_judge", judge_reason


# ──────────────────────────────────────────────
# SQLite loading
# ──────────────────────────────────────────────

def load_reviews_from_db(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    log.info(f"Loaded {len(rows)} reviews from {db_path}")
    return rows


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

class AnnotationPipeline:
    def __init__(
        self,
        db_path: str,
        output_jsonl: str = "annotated_reviews.jsonl",
        model_a: str = "gpt-4o-mini",
        model_b: str = "gemini-2.5-flash",
        judge_model: str = "gpt-4o",
        max_retries: int = 3,
        resume: bool = True,
    ):
        self.db_path = db_path
        self.output_jsonl = Path(output_jsonl)
        self.max_retries = max_retries
        self.resume = resume

        self.llm_a = ChatOpenAI(model=model_a, temperature=0.0)
        self.llm_b = ChatGoogleGenerativeAI(
            model=model_b,
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.llm_judge = ChatOpenAI(model=judge_model, temperature=0.0)

        chain_a = build_annotator_chain(self.llm_a)
        chain_b = build_annotator_chain(self.llm_b)
        self.parallel_chain = build_parallel_annotators(chain_a, chain_b)
        self.judge_chain = build_judge_chain(self.llm_judge)

    def _already_done(self) -> set:
        done = set()
        if self.resume and self.output_jsonl.exists():
            with open(self.output_jsonl, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["review_id"])
                    except Exception:
                        pass
        return done

    def _append_jsonl(self, path: Path, record: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _annotate_one(self, review_id: str, review_text: str):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(f"[{review_id}] Invoking A(gpt-4o-mini) + B(gemini) in parallel, attempt {attempt}...")
                parallel_result = await self.parallel_chain.ainvoke({"review_text": review_text})
                raw_a = parallel_result["result_a"]
                raw_b = parallel_result["result_b"]

                final, verified_by, judge_reason = await consensus_router(
                    review_text, raw_a, raw_b, self.judge_chain
                )
                log.info(
                    f"[{review_id}] Done: sentiment={final.sentiment}, "
                    f"difficulty={final.difficulty}, verified_by={verified_by}"
                )

                record = FinalRecord(
                    id=f"sample_{hashlib.md5(str(review_id).encode()).hexdigest()[:8]}",
                    review_id=str(review_id),
                    review_text=review_text,
                    sentiment=final.sentiment,
                    difficulty=final.difficulty,
                    ambiguous_flag=final.ambiguous_flag,
                    verified_by=verified_by,
                    annotator_a=raw_a,
                    annotator_b=raw_b,
                    judge_reason=judge_reason,
                    annotated_at=datetime.utcnow().isoformat(),
                )
                return record, raw_a, raw_b, final, verified_by

            except Exception as e:
                last_error = e
                log.warning(f"[{review_id}] Attempt {attempt} failed: {e}")
                await asyncio.sleep(2 ** attempt)

        log.error(f"[{review_id}] All retries exhausted: {last_error}")
        return None

    async def _worker(self, sem: asyncio.Semaphore, row: dict, results: list):
        review_id = str(row.get("id", row.get("review_id", "")))
        review_text = row.get("review_text", row.get("text", row.get("content", "")))

        if not review_text or not review_text.strip():
            log.warning(f"[{review_id}] Empty text, skipping")
            return

        async with sem:
            result = await self._annotate_one(review_id, review_text)

        if result is not None:
            record, raw_a, raw_b, final, verified_by = result
            results.append(asdict(record))

    async def run(self, limit: int = None, concurrency: int = 5):
        reviews = load_reviews_from_db(self.db_path)
        if limit:
            reviews = reviews[:limit]

        done_ids = self._already_done()
        pending = [r for r in reviews if str(r.get("id", r.get("review_id", ""))) not in done_ids]
        log.info(f"Pending: {len(pending)} reviews (skipped {len(done_ids)} already done), concurrency={concurrency}")

        sem = asyncio.Semaphore(concurrency)
        results = []
        tasks = [self._worker(sem, row, results) for row in pending]
        await asyncio.gather(*tasks)

        # 写入结果（收集后统一写，避免并发写文件冲突）
        for record in results:
            self._append_jsonl(self.output_jsonl, record)

        log.info("Pipeline complete.")
        self._print_summary()

    def _print_summary(self):
        if not self.output_jsonl.exists():
            return
        records = []
        with open(self.output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

        from collections import Counter
        print("\n" + "="*50)
        print(f"Summary ({len(records)} records)")
        print(f"  Sentiment: {dict(Counter(r.get('sentiment') for r in records))}")
        print(f"  Difficulty: {dict(Counter(r.get('difficulty') for r in records))}")
        print(f"  Verified by: {dict(Counter(r.get('verified_by') for r in records))}")
        print("="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COMP6713 Annotation Pipeline")
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-a", type=str, default="gpt-4o-mini")
    parser.add_argument("--model-b", type=str, default="gemini-2.5-flash")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max number of reviews to annotate concurrently")
    args = parser.parse_args()

    pipeline = AnnotationPipeline(
        db_path=args.db,
        model_a=args.model_a,
        model_b=args.model_b,
        judge_model=args.judge_model,
        resume=not args.no_resume,
    )
    asyncio.run(pipeline.run(limit=args.limit, concurrency=args.concurrency))