# Browser-Based Sentiment Visualization

A Chrome extension that automatically labels movie review sentiment on IMDb and Rotten Tomatoes, powered by fine-tuned Qwen3 models served from HuggingFace Spaces.

---

## Architecture

```
Chrome Extension  ──→  HuggingFace Space (FastAPI)  ──→  Qwen3 + LoRA
  (content.js)          alanwang2001/closeclaw           alanwang2001/
  (background.js)                                        qwen3-0.6B-sentiment-lora
  (popup.js)                                             qwen3-1.7B-sentiment-lora
```

---

## Supported Sites

| Site | URL Pattern |
|------|-------------|
| IMDb | `imdb.com/title/*/reviews*` |
| Rotten Tomatoes | `rottentomatoes.com/*` |

---

## Installation

### 1. Load the Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `extension/` folder from this project

### 2. Verify the Connection

Click the extension icon in the Chrome toolbar. The status dot should turn green and show **"Server online"**. If it shows offline, the HuggingFace Space may be sleeping — wait 30 seconds and try again.

---

## Usage

### Automatic Badge Labeling

1. Navigate to any supported movie review page (IMDb or Rotten Tomatoes)
2. The extension automatically scans reviews and injects sentiment badges
3. Each badge shows one of three labels:
   - **Positive** — reviewer has a favorable opinion
   - **Negative** — reviewer has an unfavorable opinion
   - **Neutral** — mixed or balanced opinion
4. Hover over a badge to see the model's reasoning

### Switching Models

1. Click the extension icon to open the popup
2. Select a model from the list:
   - **Qwen3-0.6B** — faster, lighter
   - **Qwen3-1.7B** — more accurate, slower
3. Click **Apply** and wait up to 30 seconds for the model to switch

---

## Models

| Model | HuggingFace Repo | Base |
|-------|-----------------|------|
| Qwen3-0.6B (fine-tuned) | `alanwang2001/qwen3-0.6B-sentiment-lora` | `Qwen/Qwen3-0.6B` |
| Qwen3-1.7B (fine-tuned) | `alanwang2001/qwen3-1.7B-sentiment-lora` | `Qwen/Qwen3-1.7B` |

Both models are fine-tuned with LoRA on movie review sentiment data and output labels in `\boxed{positive}` / `\boxed{negative}` / `\boxed{neutral}` format.

---

## Running the Server Locally

If you want to run the inference server on your own machine instead of using the HuggingFace Space:

```bash
cd server

# Install dependencies
pip install -r requirements.txt

# (Optional) Pre-download models
python download_models.py

# Start the server
python server.py
# Listening on http://127.0.0.1:8765
```

Then update `extension/popup.js` and `extension/background.js` to point to `http://127.0.0.1:8765`, and update `extension/manifest.json` `host_permissions` accordingly.

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
