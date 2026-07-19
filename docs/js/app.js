/* PDL live demo — main-thread UI logic. All computation happens in js/worker.js. */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  ready: false,
  trainFile: null,   // {name, buf}
  testFile: null,
  analysis: null,
  prepared: null,
  training: false,
  results: {},       // id -> result
  order: [],         // model ids in run order
  bestId: null,
};

const worker = new Worker('js/worker.js');
// Optional self-hosted Pyodide (e.g. for offline/intranet use): index.html?pyodideBase=<url>
const pyodideBase = new URLSearchParams(location.search).get('pyodideBase') || undefined;
worker.postMessage({ type: 'init', pyodideBase });

/* ---------------------------------------------------------------- runtime */
function setRuntime(msg, kind) {
  const pill = $('#runtime-pill');
  pill.className = 'pill ' + (kind || '');
  $('#runtime-msg').textContent = msg;
}

/* ------------------------------------------------------------- messaging */
worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'status':
      setRuntime(m.msg, 'busy');
      break;
    case 'ready':
      state.ready = true;
      setRuntime(`Python ${m.versions.python} · scikit-learn ${m.versions.sklearn} · pdll ${m.versions.pdll} — ready, fully in your browser`, 'ok');
      $$('.needs-runtime').forEach((b) => (b.disabled = false));
      maybeAnalyze();
      break;
    case 'analyzed':
      state.analysis = m.data;
      state.prepared = null;
      setRuntime('Runtime ready — data analyzed', 'ok');
      renderAnalysis();
      break;
    case 'prepared':
      state.prepared = m.data;
      if (!m.data.ok) {
        setRuntime('Data problem — see details below', 'warn');
        renderBlockers(m.data);
        state.training = false;
        break;
      }
      renderPrepared();
      worker.postMessage({ type: 'train' });
      break;
    case 'progress':
      onProgress(m.data);
      break;
    case 'trained':
      onTrained(m.data);
      break;
    case 'pickled':
      deliverPickle(m);
      break;
    case 'error':
      state.training = false;
      setRuntime('Error — see message below', 'warn');
      showError(`Something went wrong during "${m.where}": ${m.message}`);
      $('#train-btn').disabled = false;
      break;
  }
};

function showError(text) {
  const box = $('#error-box');
  box.hidden = false;
  box.textContent = text;
  box.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
function clearError() { $('#error-box').hidden = true; }

/* ------------------------------------------------------------ file input */
function humanSize(n) {
  if (n > 1048576) return (n / 1048576).toFixed(1) + ' MB';
  if (n > 1024) return (n / 1024).toFixed(1) + ' kB';
  return n + ' B';
}

function wireDropzone(zoneId, inputId, slot) {
  const zone = $(zoneId);
  const input = $(inputId);
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  ['dragover', 'dragenter'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('drag'); }));
  zone.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) takeFile(e.dataTransfer.files[0], slot, zone); });
  input.addEventListener('change', () => { if (input.files[0]) takeFile(input.files[0], slot, zone); });
}

async function takeFile(file, slot, zone) {
  if (file.size > 25 * 1048576) { showError(`"${file.name}" is ${humanSize(file.size)} — please keep files under 25 MB for this demo.`); return; }
  clearError();
  const buf = await file.arrayBuffer();
  state[slot] = { name: file.name, buf };
  zone.classList.add('filled');
  zone.querySelector('.dz-label').textContent = `${file.name} · ${humanSize(file.size)}`;
  if (slot === 'trainFile') maybeAnalyze();
  else if (state.trainFile) maybeAnalyze();
}

function maybeAnalyze() {
  if (!state.ready || !state.trainFile) return;
  resetResultsUI();
  setRuntime('Analyzing your data…', 'busy');
  const msg = { type: 'analyze', train: { name: state.trainFile.name, buf: state.trainFile.buf.slice(0) } };
  const transfer = [msg.train.buf];
  if (state.testFile) { msg.test = { name: state.testFile.name, buf: state.testFile.buf.slice(0) }; transfer.push(msg.test.buf); }
  worker.postMessage(msg, transfer);
}

function loadDemo(which) {
  if (!state.ready) return;
  clearError();
  resetResultsUI();
  state.trainFile = { name: which + ' (demo)', buf: null };
  state.testFile = null;
  $('#dz-train .dz-label').textContent = which + ' — built-in demo dataset';
  $('#dz-train').classList.add('filled');
  $('#dz-test .dz-label').textContent = 'Optional: drop a test file here';
  $('#dz-test').classList.remove('filled');
  worker.postMessage({ type: 'demo', which });
}

/* -------------------------------------------------------------- analysis */
function renderAnalysis() {
  const a = state.analysis;
  const card = $('#analysis-card');
  card.hidden = false;
  clearError();

  const chips = [];
  chips.push(chip(`${a.nRows ?? '?'} rows`));
  chips.push(chip(`${(a.nCols ?? 1) - 1} features`));
  if (a.target) chips.push(chip(`target: "${a.target}"`));
  if (a.task) chips.push(chip(a.task === 'classification' ? 'Classification' : 'Regression', 'accent'));
  if (a.nClasses) chips.push(chip(`${a.nClasses} classes`));
  if (a.imbalanceRatio) chips.push(chip(`imbalance ×${a.imbalanceRatio}`, a.imbalanceRatio >= 3 ? 'warn' : ''));
  if (a.missingCells) chips.push(chip(`${a.missingPct}% missing`, 'warn'));
  if (a.testRows) chips.push(chip(`test file: ${a.testRows} rows`));
  $('#data-chips').innerHTML = chips.join('');

  // target picker
  const tsel = $('#target-select');
  tsel.innerHTML = (a.columns || []).map((c) => `<option value="${escapeHtml(c)}" ${c === a.target ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('');
  $('#target-row').hidden = !a.columns;
  if (!a.targetFound && !a.target) {
    tsel.value = a.columns[a.columns.length - 1];
  }
  const ksel = $('#task-select');
  if (a.task) ksel.value = a.task;

  // class balance mini-chart
  const bal = $('#balance');
  if (a.classes && a.classes.length) {
    const max = Math.max(...a.classes.map((c) => c.count));
    bal.innerHTML = '<div class="balance-title">Class balance</div>' + a.classes.map((c) =>
      `<div class="bar-row"><span class="bar-label" title="${escapeHtml(c.label)}">${escapeHtml(c.label)}</span>
       <span class="bar-track"><span class="bar" style="width:${(100 * c.count / max).toFixed(1)}%"></span></span>
       <span class="bar-count">${c.count}</span></div>`).join('');
    bal.hidden = false;
  } else if (a.targetStats) {
    const s = a.targetStats;
    bal.innerHTML = `<div class="balance-title">Target distribution</div>
      <div class="stat-line">min ${fmt(s.min)} · mean ${fmt(s.mean)} · max ${fmt(s.max)} · std ${fmt(s.std)}</div>`;
    bal.hidden = false;
  } else bal.hidden = true;

  renderNotices('#analysis-notices', a.warnings, a.blockers);
  $('#train-btn').disabled = !!(a.blockers && a.blockers.length) || !a.target && !$('#target-select').value;
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderNotices(sel, warnings, blockers) {
  const el = $(sel);
  let html = '';
  (blockers || []).forEach((b) => { html += `<div class="notice blocker">⛔ ${escapeHtml(b)}</div>`; });
  (warnings || []).forEach((w) => { html += `<div class="notice warn">⚠️ ${escapeHtml(w)}</div>`; });
  el.innerHTML = html;
  el.hidden = !html;
}

function renderBlockers(p) {
  renderNotices('#results-notices', p.notes, p.blockers);
  $('#results-card').hidden = false;
  $('#train-btn').disabled = false;
}

$('#target-select').addEventListener('change', () => {
  worker.postMessage({ type: 'reanalyze', target: $('#target-select').value, task: null });
  resetResultsUI();
});
$('#task-select').addEventListener('change', () => {
  worker.postMessage({ type: 'reanalyze', target: $('#target-select').value, task: $('#task-select').value });
  resetResultsUI();
});

/* -------------------------------------------------------------- training */
$('#train-btn').addEventListener('click', () => {
  if (state.training) return;
  clearError();
  state.training = true;
  state.results = {};
  state.bestId = null;
  $('#train-btn').disabled = true;
  $('#results-card').hidden = false;
  $('#results-body').innerHTML = '';
  $('#insights').hidden = true;
  $('#results-notices').hidden = true;
  setRuntime('Preparing the data…', 'busy');
  worker.postMessage({
    type: 'prepare',
    config: { target: $('#target-select').value || state.analysis.target, task: $('#task-select').value || state.analysis.task },
  });
});

function renderPrepared() {
  const p = state.prepared;
  renderNotices('#results-notices', p.notes, p.blockers);
  state.order = p.models.map((m) => m.id);
  const head = p.task === 'classification'
    ? '<th>accuracy</th><th>macro F1</th>'
    : '<th>MAE</th><th>RMSE</th><th>R²</th>';
  $('#results-head').innerHTML = `<th>Model</th>${head}<th>vs base</th><th>time</th><th></th>`;
  $('#results-body').innerHTML = p.models.map((m) => `
    <tr id="row-${m.id}" class="model-row ${m.kind}">
      <td class="model-name">${m.kind === 'pdl' ? '<span class="tag tag-pdl">PDL</span>' : '<span class="tag tag-base">base</span>'} ${escapeHtml(m.name)}</td>
      <td class="cells" colspan="${p.task === 'classification' ? 2 : 3}"><span class="muted">queued…</span></td>
      <td class="delta"></td><td class="time"></td><td class="dl"></td>
    </tr>`).join('');
  $('#eval-caption').textContent =
    `Scoring on ${p.testRows} ${p.extTest ? 'rows from your test file' : 'held-out rows (80/20 split)'} · ` +
    `training on ${p.trainRows} rows × ${p.nFeatures} features — entirely in this browser tab.`;
  setRuntime('Training… fast models first', 'busy');
}

function onProgress(ev) {
  const row = $('#row-' + ev.id);
  if (!row) return;
  const task = state.prepared.task;
  const span = task === 'classification' ? 2 : 3;
  if (ev.event === 'start') {
    row.classList.add('running');
    row.querySelector('.cells').innerHTML = `<span class="spinner"></span> training…`;
    setRuntime(`Training ${ev.name}…`, 'busy');
    return;
  }
  row.classList.remove('running');
  state.results[ev.id] = ev;
  if (ev.status === 'ok') {
    const m = ev.metrics;
    const cells = task === 'classification'
      ? `<td class="num">${m.accuracy.toFixed(4)}</td><td class="num strong">${m.f1_macro.toFixed(4)}</td>`
      : `<td class="num strong">${fmt(m.mae)}</td><td class="num">${fmt(m.rmse)}</td><td class="num">${m.r2.toFixed(3)}</td>`;
    row.querySelector('.cells').outerHTML = cells;
    const deltaTd = row.querySelector('.delta');
    if (typeof ev.delta === 'number') {
      const up = ev.delta > 0.0005, down = ev.delta < -0.0005;
      deltaTd.innerHTML = `<span class="delta-badge ${up ? 'up' : down ? 'down' : 'flat'}">${up ? '▲' : down ? '▼' : '＝'} ${Math.abs(ev.delta).toFixed(4)}</span>`;
      deltaTd.title = task === 'classification' ? 'macro-F1 difference vs. the base learner' : 'MAE improvement vs. the base learner';
    }
    row.querySelector('.time').textContent = ev.seconds + 's';
    row.querySelector('.dl').innerHTML =
      `<button class="btn small" data-dl="${ev.id}" data-name="${escapeHtml(ev.name)}" title="Download the trained model as a pickle">⬇ .pkl</button>`;
  } else {
    row.classList.add('failed');
    row.querySelector('.cells').innerHTML = `<span class="fail" title="${escapeHtml(ev.trace || '')}">failed — ${escapeHtml(ev.error || 'unknown error')} (continuing with the next model)</span>`;
    row.querySelector('.time').textContent = ev.seconds + 's';
  }
}

function onTrained(summary) {
  state.training = false;
  state.bestId = summary.bestId;
  $('#train-btn').disabled = false;
  setRuntime(summary.nFail ? `Done — ${summary.nOk} models trained, ${summary.nFail} failed` : `Done — ${summary.nOk} models trained`, summary.nOk ? 'ok' : 'warn');
  if (summary.bestId) {
    const row = $('#row-' + summary.bestId);
    if (row) { row.classList.add('best'); row.querySelector('.model-name').insertAdjacentHTML('beforeend', ' <span class="tag tag-best">best</span>'); }
  }
  renderInsights(summary);
}

function renderInsights(summary) {
  const task = state.prepared.task;
  const pdl = summary.results.filter((r) => r.kind === 'pdl' && r.status === 'ok' && typeof r.delta === 'number');
  const box = $('#insights');
  if (!pdl.length) { box.hidden = true; return; }
  const wins = pdl.filter((r) => r.delta > 0.0005).length;
  const metric = task === 'classification' ? 'macro F1' : 'MAE';
  let html = `<h3>Why did PDL ${wins >= pdl.length / 2 ? 'improve' : 'struggle'} here?</h3>`;
  html += `<p class="muted">PDL improved the ${metric} of ${wins} of its ${pdl.length} base learners on this dataset. According to
    <a href="https://arxiv.org/abs/2406.20031" target="_blank" rel="noopener">Belaid, Rabus &amp; Hüllermeier (Discovery Science 2024)</a>:</p>`;
  const seen = new Set();
  const items = [];
  pdl.forEach((r) => (r.explanation || []).forEach((x) => { if (!seen.has(x)) { seen.add(x); items.push(x); } }));
  html += '<ul>' + items.map((x) => `<li>${escapeHtml(x)}</li>`).join('') + '</ul>';
  box.innerHTML = html;
  box.hidden = false;
}

/* -------------------------------------------------------------- download */
document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-dl]');
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = '…';
  worker.postMessage({ type: 'pickle', id: btn.dataset.dl, name: btn.dataset.name });
});

function deliverPickle(m) {
  const blob = new Blob([m.bytes], { type: 'application/octet-stream' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `pdl_demo_${m.id}.pkl`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  const btn = $(`[data-dl="${m.id}"]`);
  if (btn) { btn.disabled = false; btn.textContent = '⬇ .pkl'; }
  $('#snippet-code').textContent = m.snippet;
  $('#snippet-modal').showModal();
}
$('#snippet-close').addEventListener('click', () => $('#snippet-modal').close());
$('#snippet-copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('#snippet-code').textContent);
  $('#snippet-copy').textContent = 'copied ✓';
  setTimeout(() => ($('#snippet-copy').textContent = 'copy'), 1500);
});

/* --------------------------------------------------------------- helpers */
function resetResultsUI() {
  state.prepared = null;
  state.results = {};
  $('#results-card').hidden = true;
  $('#results-body').innerHTML = '';
  $('#insights').hidden = true;
}
function chip(text, cls) { return `<span class="chip ${cls || ''}">${escapeHtml(String(text))}</span>`; }
function fmt(x) {
  if (x === null || x === undefined) return '–';
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return x.toExponential(2);
  return String(Math.round(x * 10000) / 10000);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

wireDropzone('#dz-train', '#file-train', 'trainFile');
wireDropzone('#dz-test', '#file-test', 'testFile');
$$('.demo-btn').forEach((b) => b.addEventListener('click', () => loadDemo(b.dataset.demo)));
