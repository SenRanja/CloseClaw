/**
 * content.js — scans movie review pages and injects sentiment badges.
 * Each review is sent as a separate request; badges update individually as results arrive.
 */

// ── Site adapters ──────────────────────────────────────────────────────────
const ADAPTERS = [
  {
    host:           "rottentomatoes.com",
    reviewSelector: "review-card > div, review-card drawer-more > span",
  },
  {
    host:           "imdb.com",
    reviewSelector: ".ipc-html-content-inner-div",
  },
];

// ── Serial queue ───────────────────────────────────────────────────────────
// The local GPU processes one request at a time; queueing avoids pile-ups.
const queue = [];
let running = false;

function enqueue(task) {
  queue.push(task);
  if (!running) drain();
}

async function drain() {
  running = true;
  while (queue.length > 0) {
    const task = queue.shift();
    await task(); // each task updates its own badge when done
  }
  running = false;
}

// ── Sentiment fetch via background SW ─────────────────────────────────────
// Content scripts cannot fetch 127.0.0.1 from HTTPS pages (Private Network Access).
// Long-lived port avoids MV3 sendMessage timeout during slow inference.
function fetchSentiment(text) {
  return new Promise((resolve, reject) => {
    const port = chrome.runtime.connect({ name: "sentiment" });

    port.onMessage.addListener(msg => {
      port.disconnect();
      if (msg.ok) resolve(msg.data);
      else reject(new Error(msg.error ?? "server error"));
    });

    port.onDisconnect.addListener(() => {
      reject(new Error("background disconnected"));
    });

    port.postMessage({ type: "ANALYZE", text });
  });
}

// ── Badge elements ─────────────────────────────────────────────────────────
const LABEL = { positive: "Positive", negative: "Negative", neutral: "Neutral" };
const COLOR = { positive: "sentiment-positive", negative: "sentiment-negative", neutral: "sentiment-neutral" };

function makeBadge(sentiment, reasoning) {
  const el = document.createElement("span");
  el.className = `sentiment-badge ${COLOR[sentiment] ?? "sentiment-neutral"}`;
  el.title     = reasoning || sentiment;
  el.textContent = LABEL[sentiment] ?? sentiment;
  return el;
}

function makeLoadingBadge() {
  const el = document.createElement("span");
  el.className   = "sentiment-badge sentiment-loading";
  el.textContent = "Analyzing…";
  return el;
}

// ── Page scan ──────────────────────────────────────────────────────────────
function getAdapter() {
  return ADAPTERS.find(a => location.hostname.includes(a.host)) ?? null;
}

function processPage() {
  const adapter = getAdapter();
  if (!adapter) return;

  document.querySelectorAll(adapter.reviewSelector).forEach(node => {
    if (node.dataset.sentimentDone) return; // already queued
    node.dataset.sentimentDone = "1";

    const text = node.innerText.trim();
    if (text.length < 5) return;

    const loading = makeLoadingBadge();
    node.appendChild(loading);

    // One request per review — badge updates as soon as this review finishes
    enqueue(async () => {
      try {
        const result = await fetchSentiment(text);
        loading.replaceWith(makeBadge(result.sentiment, result.reasoning));
      } catch {
        loading.remove();
      }
    });
  });
}

// ── MutationObserver for SPA navigation ───────────────────────────────────
let debounce = null;
new MutationObserver(() => {
  clearTimeout(debounce);
  debounce = setTimeout(processPage, 600);
}).observe(document.body, { childList: true, subtree: true });

processPage();
