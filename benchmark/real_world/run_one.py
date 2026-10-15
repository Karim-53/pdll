"""Fit pdll and its base learner on one real dataset and print one JSON line with the outcome.

    python run_one.py <dataset name> [--config default] [--max-test 2000]

Protocol: raw data, 75/25 train/test split (stratified for classification), the test part capped at ``--max-test``
rows, the whole training part used for fitting (pdll subsamples on its own). The base learner gets the same data with
non-numeric columns ordinal-encoded, since bare scikit-learn trees do not accept strings.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEMORY_LIMIT_GB = float(os.environ.get('PDLL_BENCH_MEMORY_GB', 11))

CONFIGS = {  # PDL settings compared during the defaults study; 'default' is whatever the installed pdll does
    'default': {},
    'pairs1M_anchors1000': {'max_pairs': 1_000_000, 'max_anchors': 1000},
    'pairs250k_anchors1000': {'max_pairs': 250_000, 'max_anchors': 1000},
    'pairs250k_anchors500': {'max_pairs': 250_000, 'max_anchors': 500},
    'pairs100k_anchors500': {'max_pairs': 100_000, 'max_anchors': 500},
    'pairs100k_anchors300': {'max_pairs': 100_000, 'max_anchors': 300},
    # default ExtraTrees with larger leaves (the paper's grid searched min_samples_leaf in 1, 2, 4, 10, 21)
    'pairs1M_anchors1000_leaf5': {'max_pairs': 1_000_000, 'max_anchors': 1000, 'leaf': 5},
    'pairs250k_anchors500_leaf3': {'max_pairs': 250_000, 'max_anchors': 500, 'leaf': 3},
    'pairs250k_anchors500_leaf5': {'max_pairs': 250_000, 'max_anchors': 500, 'leaf': 5},
    'pairs250k_anchors1000_leaf5': {'max_pairs': 250_000, 'max_anchors': 1000, 'leaf': 5},
    'pairs250k_anchors500_leaf10': {'max_pairs': 250_000, 'max_anchors': 500, 'leaf': 10},
    'pairs250k_anchors500_leaf5_sqrt': {'max_pairs': 250_000, 'max_anchors': 500, 'leaf': 5, 'max_features': 'sqrt'},
    'pairs100k_anchors500_sqrt': {'max_pairs': 100_000, 'max_anchors': 500, 'leaf': 1, 'max_features': 'sqrt'},
    'pairs100k_anchors500_f03': {'max_pairs': 100_000, 'max_anchors': 500, 'leaf': 1, 'max_features': 0.3},
    # more pairs, proportionally larger leaves: about the same tree size as 250k pairs with leaf 5
    'pairs1M_anchors1000_leaf20': {'max_pairs': 1_000_000, 'max_anchors': 1000, 'leaf': 20},
    'pairs1M_anchors500_leaf20': {'max_pairs': 1_000_000, 'max_anchors': 500, 'leaf': 20},
}


class Watchdog:
    """Track peak memory; abort the process (status 'oom') above MEMORY_LIMIT_GB so the machine stays usable."""

    def __init__(self, result):
        self.result, self.peak, self.process = result, 0, psutil.Process()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            rss = self.process.memory_info().rss
            self.peak = max(self.peak, rss)
            if rss > MEMORY_LIMIT_GB * 2 ** 30:
                self.result.update(status='oom', error=f'memory above {MEMORY_LIMIT_GB} GB', peak_gb=round(rss / 2 ** 30, 2))
                print(json.dumps(self.result), flush=True)
                os._exit(3)
            time.sleep(0.05)


def encode_for_base(X_train, X_test):
    """Ordinal-encode non-numeric columns (fitted on train) for the bare base learner."""
    if not isinstance(X_train, pd.DataFrame):
        return X_train, X_test
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    non_numeric = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c])]
    encoder = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1),
                                  non_numeric)], remainder='passthrough', sparse_threshold=0.)
    if non_numeric:
        for X in (X_train, X_test):
            for c in non_numeric:
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return encoder.fit_transform(X_train).astype(float), encoder.transform(X_test).astype(float)


def scores(task, y_true, y_pred, proba=None, classes=None):
    from sklearn import metrics
    if task == 'regression':
        return {'r2': float(metrics.r2_score(y_true, y_pred)), 'mae': float(metrics.mean_absolute_error(y_true, y_pred))}
    out = {'accuracy': float(metrics.accuracy_score(y_true, y_pred)),
           'macro_f1': float(metrics.f1_score(y_true, y_pred, average='macro'))}
    if proba is not None and len(classes) == 2:
        out['roc_auc'] = float(metrics.roc_auc_score(np.asarray(y_true) == classes[1], proba[:, 1]))
    return out


def tree_nodes(model):
    estimator = getattr(model, 'estimator_', model)
    trees = getattr(estimator, 'estimators_', None)
    if trees is None:
        return None
    return int(sum(t.tree_.node_count for t in np.ravel(trees) if hasattr(t, 'tree_')))


def run(name, config, max_test, split='random'):
    import datasets
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.model_selection import train_test_split

    import pdll
    spec = next(s for s in datasets.registry() if s['name'] == name)
    result = {'dataset': name, 'source': spec['source'], 'task': spec['task'], 'config': config, 'pdll': pdll.__version__}
    watchdog = Watchdog(result)
    caught = []
    warnings.showwarning = lambda message, category, *args, **kwargs: caught.append(f'{category.__name__}: {str(message)[:160]}')
    warnings.simplefilter('always')

    t = time.perf_counter()
    try:
        X, y = datasets.load(spec)
    except Exception as error:  # noqa: BLE001
        result.update(status='load_error', error=f'{type(error).__name__}: {str(error)[:300]}')
        return result
    result['load_seconds'] = round(time.perf_counter() - t, 1)
    if isinstance(y, pd.Series) and y.isna().any():  # rows without a target cannot be used by any model
        result['dropped_missing_target'] = int(y.isna().sum())
        X, y = X[y.notna()], y[y.notna()]
    result['shape'] = list(np.shape(X))
    if isinstance(X, pd.DataFrame):
        result['non_numeric_columns'] = int(sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns))
        result['missing_values'] = int(X.isna().sum().sum())
    task = spec['task']

    try:
        stratify = None
        if task == 'classification' and np.ndim(y) == 1:
            counts = pd.Series(np.asarray(y)).value_counts()
            stratify = y if counts.min() >= 2 else None
            result['n_classes'] = int(len(counts))
        if split == 'openml-task' and 'openml_task' in spec and 'dropped_missing_target' not in result:
            import openml  # fold 0 of the official task: the split behind OpenML's public leaderboards
            train_idx, test_idx = openml.tasks.get_task(spec['openml_task'], download_splits=True, download_data=False,
                                                        download_qualities=False).get_train_test_split_indices(fold=0, repeat=0)
            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
            result['split'] = f"openml task {spec['openml_task']} fold 0"
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=stratify)
            result['split'] = 'random 75/25'
        if np.shape(X_test)[0] > max_test:
            X_test, y_test = X_test[:max_test], y_test[:max_test]
        result['n_train'], result['n_test'] = int(np.shape(X_train)[0]), int(np.shape(X_test)[0])
    except Exception as error:  # noqa: BLE001 - e.g. sparse or multi-output data
        X_train, X_test, y_train, y_test = X, X, y, y

    # pdll on the raw data: this is what is being validated
    params = dict(CONFIGS[config], random_state=0)
    leaf, max_features = params.pop('leaf', None), params.pop('max_features', None)
    if leaf is not None:
        extra = {} if max_features is None else {'max_features': max_features}
        params['estimator'] = (ExtraTreesClassifier(class_weight='balanced', min_samples_leaf=leaf, n_jobs=-1, **extra) if task == 'classification'
                               else ExtraTreesRegressor(min_samples_leaf=leaf, n_jobs=-1, **extra))
    model = (pdll.PairwiseDifferenceClassifier if task == 'classification' else pdll.PairwiseDifferenceRegressor)(**params)
    try:
        t = time.perf_counter()
        model.fit(X_train, y_train)
        result['pdl_fit_seconds'] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test) if task == 'classification' else None
        result['pdl_predict_seconds'] = round(time.perf_counter() - t, 2)
        result['pdl'] = scores(task, y_test, pred, proba, getattr(model, 'classes_', None))
        result['pdl_tree_nodes'] = tree_nodes(model)
        result['n_anchors'] = int(len(model.X_anchors_))
        result['status'] = 'ok'
    except Exception as error:  # noqa: BLE001
        result.update(status='pdl_error', error=f'{type(error).__name__}: {str(error)[:400]}',
                      traceback=traceback.format_exc().splitlines()[-6:])
    result['pdl_peak_gb'] = round(watchdog.peak / 2 ** 30, 2)
    del model

    # the base learner alone, for reference
    if result['status'] == 'ok':
        try:
            Xb_train, Xb_test = encode_for_base(X_train.copy() if hasattr(X_train, 'copy') else X_train,
                                                X_test.copy() if hasattr(X_test, 'copy') else X_test)
            base = (ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=0) if task == 'classification'
                    else ExtraTreesRegressor(n_jobs=-1, random_state=0))
            yb_train, yb_test = np.asarray(y_train), np.asarray(y_test)
            if task == 'classification':  # bare forests with class_weight reject some categorical targets: use codes
                labels = np.unique(np.concatenate([yb_train, yb_test]).astype(str))
                yb_train, yb_test = np.searchsorted(labels, yb_train.astype(str)), np.searchsorted(labels, yb_test.astype(str))
            t = time.perf_counter()
            base.fit(Xb_train, yb_train)
            pred = base.predict(Xb_test)
            proba = base.predict_proba(Xb_test) if task == 'classification' else None
            result['base_seconds'] = round(time.perf_counter() - t, 2)
            result['base'] = scores(task, yb_test, pred, proba, getattr(base, 'classes_', None))
        except Exception as error:  # noqa: BLE001
            result['base_error'] = f'{type(error).__name__}: {str(error)[:200]}'
    result['warnings'] = sorted(set(caught))[:10]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--config', default='default', choices=sorted(CONFIGS))
    parser.add_argument('--max-test', type=int, default=2000)
    parser.add_argument('--split', default='random', choices=['random', 'openml-task'])
    args = parser.parse_args()
    print(json.dumps(run(args.name, args.config, args.max_test, args.split)), flush=True)
