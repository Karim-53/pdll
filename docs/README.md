# PDL live demo (GitHub Pages)

A zero-backend proof of concept for [pdll](https://github.com/Karim-53/pdll): visitors drop a
`train` file (and optionally a `test` file) with a column named `target`, and the page benchmarks
scikit-learn baselines against **PDL(DecisionTree)**, **PDL(ExtraTrees)** and **PDL(RandomForest)**
— classification or regression, detected automatically.

Everything runs **inside the visitor's browser**: [Pyodide](https://pyodide.org) provides Python,
NumPy, pandas and scikit-learn compiled to WebAssembly, and the demo imports the *exact* `pdll`
source of this repository (`docs/py/pdll/`, kept as a copy of `/pdll/`). No data ever leaves the
visitor's machine; the trained models can be downloaded as regular Python pickles and loaded
anywhere with `pip install pdll scikit-learn`.

## Enable it

Repository → Settings → Pages → *Deploy from a branch* → branch `main`, folder `/docs`.
The demo is then served at `https://karim-53.github.io/pdll/`.

## What the page handles

- **File formats**: CSV (any common separator), TSV, TXT, XLSX, JSON.
- **Task detection**: numeric target with many distinct values → regression, otherwise
  classification; the visitor can override both the task and the target column.
- **Quick-demo cap**: training uses at most **599 rows** (stratified random sample, explained
  in the UI). A second cap keeps the n² pairwise matrix inside browser memory when there are
  many features — all models then train on the same subsample so the comparison stays fair.
- **Messy data**: categorical features are ordinal-encoded, datetimes become numeric, ID-like /
  constant / all-missing columns are dropped, missing values are median-imputed — except for
  HistGradientBoosting, which consumes missing values natively. Every decision is surfaced as a
  notice, with pointers to the paper's findings (PDL is sensitive to class imbalance, missing
  values, and datasets beyond ~2000 rows).
- **Robust runs**: models execute fastest-first and a failing model never stops the run.

## Files

```
docs/
├── index.html        # the page
├── css/style.css     # styling (light + dark)
├── js/app.js         # UI logic (main thread)
├── js/worker.js      # Web Worker hosting Pyodide
└── py/
    ├── trainer.py    # data loading, checks, training, explanations, pickling
    └── pdll/         # copy of this repo's pdll package, imported by the worker
```

`docs/py/trainer.py` is plain Python — run `python docs/py/trainer.py` locally for a smoke test
of the whole pipeline on the three built-in demo datasets.

## Self-hosting the runtime

By default the Pyodide runtime (~40 MB, cached by the browser after the first visit) is fetched
from the jsDelivr CDN. For offline or intranet use, mirror the Pyodide `full` distribution and
open the page as `index.html?pyodideBase=https://your-host/pyodide/`.
