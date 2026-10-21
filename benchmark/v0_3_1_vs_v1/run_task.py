"""One OpenML task, one pdll version: official 10-fold splits, default settings.

    python -I run_task.py --version now|0.3.1 [--pdll031 DIR] --out results.jsonl TASK_ID

pdll now gets the raw DataFrame exactly as OpenML passes it (X.iloc[train]), as in pdll's
benchmark/real_world/openml_leaderboard.py. pdll 0.3.1 is scored two ways on the same folds:
  raw      the same raw DataFrame and categorical target a user would pass;
  encoded  its best case: numpy arrays, categorical features ordinal-encoded (NaN kept), integer labels,
           i.e. what 1.0 does internally, which sidesteps 0.3.1's dtype, pandas-index and categorical-label bugs.
"""
import argparse
import json
import os
import sys
import threading
import time
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--version', choices=['now', '0.3.1'], required=True)
parser.add_argument('--pdll031')
parser.add_argument('--out', required=True)
parser.add_argument('--mem-gb', type=float, default=10.0)
parser.add_argument('task', type=int)
args = parser.parse_args()
if args.version == '0.3.1':
    sys.path.insert(0, args.pdll031)

import numpy as np
import openml
import pandas as pd
import pdll
import psutil
from pdll import PairwiseDifferenceClassifier
from sklearn.metrics import accuracy_score

expected = '0.3.1' if args.version == '0.3.1' else '1.0.0'
assert pdll.__version__ == expected, (pdll.__version__, pdll.__file__)

task = openml.tasks.get_task(args.task)
dataset = task.get_dataset()
result = {'task': args.task, 'dataset': dataset.name, 'rows': int(dataset.qualities['NumberOfInstances']),
          'pdll': pdll.__version__}


def write(record):
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


peak = [0.0]


def watchdog():
    proc = psutil.Process()
    while True:
        info = proc.memory_info()  # committed (private) memory on Windows: RSS stays low while the pagefile fills up
        rss = max(info.rss, getattr(info, 'private', 0)) / 2**30
        peak[0] = max(peak[0], rss)
        if rss > args.mem_gb:
            write({**result, 'status': 'out_of_memory', 'error': f'RSS above {args.mem_gb} GB', 'peak_gb': round(rss, 2)})
            os._exit(3)
        time.sleep(0.2)


def encode(X):
    """Ordinal-encode non-numeric columns (NaN stays NaN) and return a float array."""
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        s = X[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_bool_dtype(s):
            codes = s.astype('category').cat.codes.astype(float)
            out[col] = codes.where(codes >= 0, np.nan)
        else:
            out[col] = s.astype(float)
    return out.to_numpy()


def error_text(e):
    text = str(e).splitlines()[0][:200] if str(e) else ''
    return f'{type(e).__name__}: {text}'


threading.Thread(target=watchdog, daemon=True).start()
X, y = task.get_X_and_y(dataset_format='dataframe')
start = time.time()
warnings.simplefilter('ignore')

if args.version == 'now':
    acc, status, error = [], 'ok', None
    for fold in range(10):
        train, test = task.get_train_test_split_indices(fold=fold, repeat=0)
        try:
            model = PairwiseDifferenceClassifier(random_state=0).fit(X.iloc[train], y.iloc[train])
            acc.append(accuracy_score(np.asarray(y.iloc[test]), np.asarray(model.predict(X.iloc[test]))))
        except Exception as e:
            status, error = 'error', error_text(e)
            break
    result.update(status=status, error=error, accuracy_10fold=round(float(np.mean(acc)), 4) if status == 'ok' else None)
else:
    X_enc, y_codes = encode(X), y.cat.codes.to_numpy()
    acc_raw, raw_error, acc_enc, enc_error = [], None, [], None
    for fold in range(10):
        train, test = task.get_train_test_split_indices(fold=fold, repeat=0)
        if raw_error is None:
            try:
                model = PairwiseDifferenceClassifier().fit(X.iloc[train], y.iloc[train])
                acc_raw.append(accuracy_score(np.asarray(y.iloc[test]).astype(str), np.asarray(model.predict(X.iloc[test])).astype(str)))
            except Exception as e:
                raw_error = error_text(e)
        if enc_error is None:
            try:
                model = PairwiseDifferenceClassifier().fit(X_enc[train], y_codes[train])
                acc_enc.append(accuracy_score(y_codes[test], np.asarray(model.predict(X_enc[test])).astype(y_codes.dtype)))
            except Exception as e:
                enc_error = error_text(e)
        if raw_error and enc_error:
            break
    result.update(status='ok' if enc_error is None else 'error',
                  raw_accuracy_10fold=round(float(np.mean(acc_raw)), 4) if raw_error is None else None, raw_error=raw_error,
                  encoded_accuracy_10fold=round(float(np.mean(acc_enc)), 4) if enc_error is None else None, encoded_error=enc_error)

result.update(seconds=round(time.time() - start, 1), peak_gb=round(peak[0], 2))
write(result)
