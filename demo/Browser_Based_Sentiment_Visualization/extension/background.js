// background.js — service worker
// Uses long-lived port connections to avoid MV3 message channel timeout
// during slow inference (Qwen3 can take several seconds per review).

chrome.runtime.onConnect.addListener(port => {
  if (port.name !== "sentiment") return;

  port.onMessage.addListener(msg => {
    if (msg.type !== "ANALYZE") return;

    fetch("http://127.0.0.1:8765/analyze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text: msg.text }),
    })
      .then(r => r.json())
      .then(data => port.postMessage({ ok: true, data }))
      .catch(err => port.postMessage({ ok: false, error: err.message }));
  });
});
