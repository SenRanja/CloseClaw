// background.js — service worker
// Uses long-lived port connections to avoid MV3 message channel timeout
// during slow inference (Qwen3 can take several seconds per review).

chrome.runtime.onConnect.addListener(port => {
  if (port.name !== "sentiment") return;

  port.onMessage.addListener(msg => {
    if (msg.type !== "ANALYZE") return;

    fetch("https://alanwang2001-closeclaw.hf.space/analyze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text: msg.text }),
    })
      .then(r => r.json())
      .then(data => port.postMessage({ ok: true, data }))
      .catch(err => port.postMessage({ ok: false, error: err.message }));
  });
});
