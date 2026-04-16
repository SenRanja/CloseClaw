"""
FastAPI inference server for sentiment analysis.
Supports dynamic switching between three fine-tuned LoRA models.

Run: python server.py
Listens on http://127.0.0.1:8765
"""

import re
import os
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "E:/hf_cache/hub")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Model registry ────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "qwen3-0.6B": {
        "label":    "Qwen3-0.6B (Fine-tuned)",
        "base":     "Qwen/Qwen3-0.6B",
        "lora":     os.path.join(PROJECT_DIR, "qwen3-0.6B", "lora", "sft"),
    },
    "qwen3-1.7B": {
        "label":    "Qwen3-1.7B (Fine-tuned)",
        "base":     "Qwen/Qwen3-1.7B",
        "lora":     os.path.join(PROJECT_DIR, "qwen3-1.7B", "lora", "sft"),
    },
}

DEFAULT_MODEL = "qwen3-0.6B"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given movie review into one of three categories:\n"
    "- positive: the reviewer expresses a favorable opinion of the movie.\n"
    "- negative: the reviewer expresses an unfavorable opinion of the movie.\n"
    "- neutral: the reviewer expresses a mixed or balanced opinion with no clear positive or negative leaning.\n"
    "First explain your reasoning, then put your final answer in \\boxed{}, "
    "for example \\boxed{positive}."
)

VALID_LABELS = {"positive", "negative", "neutral"}

# ── Model state ───────────────────────────────────────────────────────────────

device           = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer        = None
model            = None
current_model_name = None
model_status     = "loading"   # loading | ready | switching | error
_executor        = ThreadPoolExecutor(max_workers=1)


def _do_load(name: str):
    """Blocking model load — runs in thread executor."""
    global tokenizer, model, current_model_name, model_status

    cfg = MODELS[name]
    print(f"\n[load] {name}  base={cfg['base']}  lora={cfg['lora']}")

    # Unload existing model
    if model is not None:
        model = None
        tokenizer = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["lora"] if os.path.exists(os.path.join(cfg["lora"], "tokenizer.json"))
                    else cfg["base"],
        trust_remote_code=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"],
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": 0} if device == "cuda" else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, cfg["lora"])
    model.eval()
    if device == "cpu":
        model.to(device)

    current_model_name = name
    model_status       = "ready"
    print(f"[load] done — {name} ready on {device}")


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class PrivateNetworkMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

app.add_middleware(PrivateNetworkMiddleware)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _do_load, DEFAULT_MODEL)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": model_status, "device": device, "model": current_model_name}


@app.get("/models")
def list_models():
    return [
        {
            "id":      mid,
            "label":   cfg["label"],
            "current": mid == current_model_name,
        }
        for mid, cfg in MODELS.items()
    ]


class SwitchRequest(BaseModel):
    model: str


@app.post("/model")
async def switch_model(req: SwitchRequest):
    global model_status

    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")
    if req.model == current_model_name and model_status == "ready":
        return {"model": current_model_name, "status": "ready"}

    model_status = "switching"
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(_executor, _do_load, req.model)
    except Exception as e:
        model_status = "error"
        raise HTTPException(status_code=500, detail=str(e))

    return {"model": current_model_name, "status": "ready"}


# ── Inference ─────────────────────────────────────────────────────────────────

def build_prompt(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Classify the sentiment of this movie review:\n\n{text[:512]}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_label(text: str) -> str:
    m = re.search(r"\\boxed\{(\w+)\}", text)
    if m and m.group(1).lower() in VALID_LABELS:
        return m.group(1).lower()
    for label in VALID_LABELS:
        if label in text.lower():
            return label
    return "neutral"


class ReviewRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(req: ReviewRequest):
    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Model is {model_status}")

    prompt  = build_prompt(req.text)
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    generated  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    sentiment  = parse_label(generated)
    reasoning  = re.sub(r"\\boxed\{\w+\}", "", generated).strip()

    return {"sentiment": sentiment, "reasoning": reasoning}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
