# 7. Browser-Based Real-Time Sentiment Visualization

## 7.1 System Overview

To bridge the gap between model inference and end-user experience, we developed a
Chrome browser extension that automatically annotates movie reviews on Rotten Tomatoes
and IMDb with sentiment labels in real time. When a user browses a review page, each
review is analyzed by a locally hosted fine-tuned Qwen3 model and tagged with a
color-coded badge — **Positive**, **Negative**, or **Neutral** — inline with the
original text, without any manual interaction.

---

## 7.2 Architecture

The system consists of two decoupled components that communicate over localhost HTTP.

```
┌─────────────────────────────────────────────────┐
│              Chrome Extension (MV3)             │
│                                                 │
│  content.js          background.js              │
│  ─ DOM scanning    ←─ chrome.runtime.connect ─→ │
│  ─ Badge injection   ─ Fetch proxy              │
│  ─ Serial queue      ─ Localhost relay          │
└───────────────────────┬─────────────────────────┘
                        │ HTTP POST /analyze
                        ▼
┌─────────────────────────────────────────────────┐
│           Local Inference Server (Python)       │
│                                                 │
│  FastAPI · Uvicorn · 127.0.0.1:8765             │
│  ─ AutoModelForCausalLM (Qwen3 base)            │
│  ─ PeftModel (LoRA SFT adapter)                 │
│  ─ /analyze  /models  /model  /health           │
└─────────────────────────────────────────────────┘
```

---

## 7.3 Key Design Decisions

### 7.3.1 LoRA Adapter Loading

Each fine-tuned model is loaded as a LoRA adapter layered on top of the corresponding
Qwen3 base model via `PeftModel.from_pretrained`, keeping adapter storage at 77–133 MB
per variant.

### 7.3.2 Chrome Private Network Access Bypass

Chrome's Private Network Access (PNA) policy blocks HTTPS pages from directly
fetching `127.0.0.1`. We route all inference requests through the background service
worker, which is exempt from PNA checks. A long-lived `chrome.runtime.connect` port
is used to prevent the MV3 service worker from being terminated during slow inference.

### 7.3.3 Serial Request Queue

Since the local GPU processes one sequence at a time, the content script maintains a
client-side serial queue: each review fires its request only after the previous one
returns, while its placeholder badge (`Analyzing…`) updates independently upon
completion.

### 7.3.4 Dynamic Model Switching

The server exposes a `POST /model` endpoint that unloads the current model before
loading the selected one. A popup dashboard lists available fine-tuned variants
(Qwen3-0.6B and Qwen3-1.7B) with live status indicators, allowing users to switch
models without restarting the server.

---

## 7.4 Site Adapter Design

Review extraction is driven by a per-site CSS selector configuration. The
MutationObserver pattern handles React-driven dynamic loading across all tabs.

| Site | Review Selector | Dynamic Loading |
|------|----------------|-----------------|
| Rotten Tomatoes | `review-card > div,` `review-card drawer-more > span` | React SPA (MutationObserver) |
| IMDb | `.ipc-html-content-inner-div` | React SPA (MutationObserver) |

---

## 7.5 Inference Prompt Format

Inference uses the same ChatML prompt format as training. The model outputs a
reasoning chain followed by a boxed label (`\boxed{positive}`, `\boxed{negative}`,
or `\boxed{neutral}`), which is extracted by regex and displayed as a badge; the
reasoning chain is surfaced as a tooltip.


## Models

| Model | HuggingFace Repo | Base |
|-------|-----------------|------|
| Qwen3-0.6B (fine-tuned) | `alanwang2001/qwen3-0.6B-sentiment-lora` | `Qwen/Qwen3-0.6B` |
| Qwen3-1.7B (fine-tuned) | `alanwang2001/qwen3-1.7B-sentiment-lora` | `Qwen/Qwen3-1.7B` |

Both models are fine-tuned with LoRA on movie review sentiment data and output labels in `\boxed{positive}` / `\boxed{negative}` / `\boxed{neutral}` format.

---

## Project Structure

```
Browser_Based_Sentiment_Visualization/
├── Dockerfile              # HuggingFace Space container config
├── extension/              # Chrome extension
│   ├── manifest.json
│   ├── popup.html/js       # Extension popup UI
│   ├── background.js       # Service worker (proxies API calls)
│   ├── content.js          # Injects badges into review pages
│   └── badge.css           # Badge styles
└── server/                 # FastAPI inference server
    ├── server.py
    ├── requirements.txt
    └── download_models.py  # Pre-download script for local use
```
