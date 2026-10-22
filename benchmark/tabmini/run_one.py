"""Score one model on one TabMini dataset with TabMini's published protocol.

    python run_one.py --out results/tabmini.jsonl parity5 MODEL [--n-jobs 12]

Protocol of TabMini (Knauer et al. 2024, https://github.com/RicardoKnauer/TabMini at 385961e), reproduced exactly:
the PMLB file, NaN rows dropped, rows shuffled with ``sample(frac=1, random_state=42)``, positive class = the largest
label, ``StratifiedKFold(3)`` without further shuffling, ROC AUC per test fold, averaged over the 3 folds.
Every model is scored on ``predict_proba``. The models are those of the small-data benchmark (``../small_data/models.py``).

Appends one JSON line: mean ROC AUC, accuracy, macro-F1 and log loss over the folds, the per-fold AUC, fit and predict
seconds and peak memory.
"""
import argparse
import json
import os
import platform
import sys
import threading
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'small_data'))  # models.py
from models import check_environment, make_model  # noqa: E402

PMLB_PIN = '872ec0a1f99ca0b55a8c8c5491da09a7da66ce13'  # last PMLB datasets/ commit before 9 of the 44 were renamed _deprecated_*
PMLB_CACHE = os.path.join(HERE, 'pmlb_cache')


def load(name):
    """(X, y) exactly as ``tabmini.load_dataset`` and the label encoding of ``tabmini.compare``."""
    path = os.path.join(PMLB_CACHE, f'{name}.tsv.gz')
    if not os.path.exists(path):
        os.makedirs(PMLB_CACHE, exist_ok=True)
        urllib.request.urlretrieve(f'https://github.com/EpistasisLab/pmlb/raw/{PMLB_PIN}/datasets/{name}/{name}.tsv.gz', path)
    frame = pd.read_csv(path, sep='\t', compression='gzip').dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    return frame.drop(columns='target'), (frame['target'] == frame['target'].max()).astype(int).to_numpy()


def folds(y):
    return list(StratifiedKFold(3).split(np.zeros(len(y)), y))


def positive_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, list(np.asarray(model.classes_, dtype=int)).index(1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-jobs', type=int, default=12)
    args = parser.parse_args()
    check_environment([args.model])

    import pdll
    import sklearn

    peak = [0.]

    def watch():
        proc = psutil.Process()
        while True:
            peak[0] = max(peak[0], proc.memory_info().rss / 2**30)
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()
    X, y = load(args.dataset)
    record = {'dataset': f'tabmini:{args.dataset}', 'name': args.dataset, 'cv': 'tabmini', 'rows': len(X), 'features': X.shape[1],
              'classes': 2, 'minority': int(np.bincount(y).min()), 'model': args.model,
              'pdll': pdll.__version__, 'sklearn': sklearn.__version__, 'python': platform.python_version()}
    auc, acc, f1, ll, fit_s, predict_s = [], [], [], [], 0., 0.
    try:
        # OpenMP/BLAS pools (HistGradientBoosting, LightGBM, XGBoost) get the same thread budget as n_jobs
        with warnings.catch_warnings(), threadpool_limits(limits=args.n_jobs):
            warnings.simplefilter('ignore')
            for train, test in folds(y):
                model = make_model(args.model, X.iloc[train], args.n_jobs)
                start = time.perf_counter()
                model.fit(X.iloc[train], y[train])
                fit_s += time.perf_counter() - start
                start = time.perf_counter()
                p1 = positive_proba(model, X.iloc[test])
                prediction = np.asarray(model.predict(X.iloc[test])).astype(int)
                predict_s += time.perf_counter() - start
                auc.append(roc_auc_score(y[test], p1))
                acc.append(accuracy_score(y[test], prediction))
                f1.append(f1_score(y[test], prediction, average='macro'))
                ll.append(log_loss(y[test], np.clip(p1, 1e-15, 1 - 1e-15), labels=[0, 1]))
        record.update(status='ok', roc_auc=round(float(np.mean(auc)), 5), accuracy=round(float(np.mean(acc)), 5),
                      macro_f1=round(float(np.mean(f1)), 5), log_loss=round(float(np.mean(ll)), 5),
                      fold_auc=[round(float(a), 5) for a in auc])
    except Exception as e:  # noqa: BLE001 - a failing model is a result, not a crash of the benchmark
        record.update(status='error', error=f'{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ""}', folds_done=len(auc))
    record.update(fit_seconds=round(fit_s, 2), predict_seconds=round(predict_s, 2), peak_gb=round(peak[0], 2))
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
