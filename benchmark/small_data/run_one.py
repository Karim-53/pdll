"""Score one model on one OpenML task: official 10-fold split, raw DataFrame, default settings.

    python run_one.py --out results/cc18_small.jsonl TASK_ID MODEL [--n-jobs 4]

Appends one JSON line: mean accuracy, macro-F1 and ROC AUC (one-vs-rest macro for multiclass) over the 10 folds,
plus fit and predict seconds and peak memory.
"""
import argparse
import json
import platform
import threading
import time
import warnings

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from models import check_environment, make_model


def full_proba(model, X, n_classes):
    """predict_proba with one column per class of the whole dataset, even if a training fold missed one."""
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out


def roc_auc(y, proba):
    try:
        if proba.shape[1] == 2:
            return roc_auc_score(y, proba[:, 1])
        present = np.unique(y)
        sub = proba[:, present] / np.clip(proba[:, present].sum(axis=1, keepdims=True), 1e-12, None)
        return roc_auc_score(y, sub, multi_class='ovr', average='macro', labels=present)
    except ValueError:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=int)
    parser.add_argument('model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()
    check_environment([args.model])

    import openml
    import pdll
    import sklearn

    peak = [0.]

    def watch():
        proc = psutil.Process()
        while True:
            peak[0] = max(peak[0], proc.memory_info().rss / 2**30)
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()
    task = openml.tasks.get_task(args.task)
    dataset = task.get_dataset()
    X, y = task.get_X_and_y(dataset_format='dataframe')
    y = LabelEncoder().fit_transform(y)
    n_classes = int(y.max()) + 1
    record = {'task': args.task, 'dataset': dataset.name, 'rows': len(X), 'features': X.shape[1], 'classes': n_classes,
              'model': args.model, 'pdll': pdll.__version__, 'sklearn': sklearn.__version__, 'python': platform.python_version()}
    acc, f1, auc, fit_s, predict_s = [], [], [], 0., 0.
    try:
        # OpenMP/BLAS pools (HistGradientBoosting, LightGBM, XGBoost) get the same thread budget as n_jobs
        with warnings.catch_warnings(), threadpool_limits(limits=args.n_jobs):
            warnings.simplefilter('ignore')
            for fold in range(10):
                train, test = task.get_train_test_split_indices(fold=fold, repeat=0)
                model = make_model(args.model, X.iloc[train], args.n_jobs)
                start = time.perf_counter()
                model.fit(X.iloc[train], y[train])
                fit_s += time.perf_counter() - start
                start = time.perf_counter()
                proba = full_proba(model, X.iloc[test], n_classes)
                prediction = np.asarray(model.predict(X.iloc[test]))
                predict_s += time.perf_counter() - start
                acc.append(accuracy_score(y[test], prediction))
                f1.append(f1_score(y[test], prediction, average='macro'))
                auc.append(roc_auc(y[test], proba))
        record.update(status='ok', accuracy=round(float(np.mean(acc)), 4), macro_f1=round(float(np.mean(f1)), 4),
                      roc_auc=round(float(np.nanmean(auc)), 4))
    except Exception as e:  # noqa: BLE001 - a failing model is a result, not a crash of the benchmark
        record.update(status='error', error=f'{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ""}', folds_done=len(acc))
    record.update(fit_seconds=round(fit_s, 2), predict_seconds=round(predict_s, 2), peak_gb=round(peak[0], 2))
    with open(args.out, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
