/* Web Worker: hosts the Pyodide runtime and runs all training off the UI thread. */
'use strict';

const PYODIDE_VERSION = '0.26.4';
let PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodide = null;
let trainer = null;
let excelReady = false;

function status(msg) { self.postMessage({ type: 'status', msg }); }

async function init(baseOverride) {
  if (baseOverride) PYODIDE_URL = baseOverride; // e.g. ?pyodideBase=… for self-hosting
  importScripts(PYODIDE_URL + 'pyodide.js');
  status('Downloading the Python runtime (~15 MB, cached after first visit)…');
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status('Loading NumPy, pandas and scikit-learn (~30 MB, cached)…');
  await pyodide.loadPackage(['numpy', 'pandas', 'scipy', 'scikit-learn']);

  status('Installing pdll (the exact source from this repository)…');
  const files = {
    '/app/pdll/__init__.py': '../py/pdll/__init__.py',
    '/app/pdll/_pairwise.py': '../py/pdll/_pairwise.py',
    '/app/trainer.py': '../py/trainer.py',
  };
  pyodide.FS.mkdirTree('/app/pdll');
  for (const [dst, src] of Object.entries(files)) {
    const r = await fetch(new URL(src, self.location.href));
    if (!r.ok) throw new Error(`Failed to fetch ${src} (${r.status})`);
    pyodide.FS.writeFile(dst, await r.text());
  }
  pyodide.runPython(`
import sys
sys.path.insert(0, '/app')
import trainer
`);
  trainer = pyodide.globals.get('trainer');
  pyodide.globals.set('_js_emit', (s) => self.postMessage({ type: 'progress', data: JSON.parse(s) }));
  pyodide.runPython('trainer.set_progress_callback(_js_emit)');

  const versions = pyodide.runPython(`
import json, sys, sklearn, pandas, numpy, pdll
json.dumps({'python': sys.version.split()[0], 'sklearn': sklearn.__version__,
            'pandas': pandas.__version__, 'numpy': numpy.__version__,
            'pdll': getattr(pdll, '__version__', 'dev'), 'pyodide': '${PYODIDE_VERSION}'})
`);
  self.postMessage({ type: 'ready', versions: JSON.parse(versions) });
}

async function ensureExcel(names) {
  const needsExcel = names.some((n) => n && /\.(xlsx|xlsm|xls)$/i.test(n));
  if (needsExcel && !excelReady) {
    status('Loading the Excel reader (openpyxl)…');
    await pyodide.loadPackage('openpyxl');
    excelReady = true;
  }
}

const handlers = {
  async init(msg) { await init(msg.pyodideBase); },

  async analyze(msg) {
    await ensureExcel([msg.train.name, msg.test && msg.test.name]);
    status('Reading your data…');
    const trainBuf = new Uint8Array(msg.train.buf);
    let res;
    if (msg.test) {
      res = trainer.load_files(trainBuf, msg.train.name, new Uint8Array(msg.test.buf), msg.test.name);
    } else {
      res = trainer.load_files(trainBuf, msg.train.name);
    }
    self.postMessage({ type: 'analyzed', data: JSON.parse(res) });
  },

  async demo(msg) {
    status('Generating the demo dataset…');
    const res = trainer.load_demo(msg.which);
    self.postMessage({ type: 'analyzed', data: JSON.parse(res) });
  },

  async reanalyze(msg) {
    const res = trainer.analyze(msg.target, msg.task);
    self.postMessage({ type: 'analyzed', data: JSON.parse(res) });
  },

  async prepare(msg) {
    status('Encoding features and building the train/test split…');
    const res = trainer.prepare(JSON.stringify(msg.config));
    self.postMessage({ type: 'prepared', data: JSON.parse(res) });
  },

  async train() {
    const res = trainer.train_all();
    self.postMessage({ type: 'trained', data: JSON.parse(res) });
  },

  async pickle(msg) {
    const proxy = trainer.get_model_pickle(msg.id);
    const bytes = proxy.toJs();
    proxy.destroy();
    const snippet = trainer.get_load_snippet(msg.id);
    self.postMessage({ type: 'pickled', id: msg.id, name: msg.name, bytes, snippet }, [bytes.buffer]);
  },
};

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    await handlers[msg.type](msg);
  } catch (err) {
    let text = String(err && err.message ? err.message : err);
    if (text.length > 1500) text = '…' + text.slice(-1500); // keep the tail: Python puts the real error last
    self.postMessage({ type: 'error', where: msg.type, message: text });
  }
};
