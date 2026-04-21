const API = 'https://alanwang2001-closeclaw.hf.space';
let currentModel = null;
let selectedModel = null;

async function refresh() {
  try {
    const [health, models] = await Promise.all([
      fetch(API + '/health').then(r => r.json()),
      fetch(API + '/models').then(r => r.json()),
    ]);
    currentModel  = health.model;
    selectedModel = selectedModel ?? currentModel;
    setStatus(health.status, health.model);
    renderModels(models);
  } catch {
    setStatus('error', null);
  }
}

function setStatus(status, modelName) {
  const dot  = document.getElementById('dot');
  const text = document.getElementById('statusText');
  dot.className = 'dot ' + status;
  const labels = {
    ready:     'Server online',
    loading:   'Loading model…',
    switching: 'Switching model…',
    error:     'Server offline',
  };
  text.textContent = (labels[status] ?? status) + (modelName ? '  ·  ' + modelName : '');
}

function renderModels(models) {
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  models.forEach(function(m) {
    var item = document.createElement('label');
    item.className = 'model-item' + (m.id === selectedModel ? ' selected' : '');

    var radio = document.createElement('input');
    radio.type    = 'radio';
    radio.name    = 'model';
    radio.value   = m.id;
    radio.checked = m.id === selectedModel;
    radio.addEventListener('change', function() {
      selectedModel = m.id;
      document.querySelectorAll('.model-item').forEach(function(el) {
        el.classList.remove('selected');
      });
      item.classList.add('selected');
      updateApplyBtn();
    });

    var info  = document.createElement('div');
    info.className = 'model-info';

    var name  = document.createElement('div');
    name.className   = 'model-name';
    name.textContent = m.label;

    var badge = document.createElement('div');
    badge.className   = 'model-badge ' + (m.current ? 'active' : 'idle');
    badge.textContent = m.current ? 'Active' : 'Idle';

    info.appendChild(name);
    info.appendChild(badge);
    item.appendChild(radio);
    item.appendChild(info);
    list.appendChild(item);
  });
  updateApplyBtn();
}

function updateApplyBtn() {
  var btn  = document.getElementById('applyBtn');
  var note = document.getElementById('applyNote');
  var same = selectedModel === currentModel;
  btn.disabled     = same;
  note.textContent = same ? 'This model is already active' : '';
}

document.getElementById('applyBtn').addEventListener('click', async function() {
  var btn  = document.getElementById('applyBtn');
  var note = document.getElementById('applyNote');
  btn.disabled     = true;
  btn.textContent  = 'Switching…';
  note.textContent = 'This may take up to 30s';
  setStatus('switching', selectedModel);
  try {
    var res  = await fetch(API + '/model', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ model: selectedModel }),
    });
    var data = await res.json();
    currentModel     = data.model;
    btn.textContent  = 'Apply';
    note.textContent = '✓ Model switched';
    await refresh();
  } catch (e) {
    btn.textContent  = 'Apply';
    note.textContent = 'Switch failed — server error';
    setStatus('error', null);
  }
});

refresh();
