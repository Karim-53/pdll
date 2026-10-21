"""End-to-end benchmark: PDC and PDR versus their base learner on small offline datasets.

Mirrors the paper protocol on a CI budget: 5-fold CV (stratified for classification), macro-F1 for PDC and R² for
PDR, base learner with default parameters versus PDL wrapping the same learner (PDC with
``class_weight='balanced'``), fixed seeds.

    python tests/e2e/benchmark.py            # print the results
    python tests/e2e/update_baseline.py      # store them as the new baseline (deliberate step)
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_moons,
    make_regression,
)
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tests/ for titanic_like
BASELINE_PATH = os.path.join(HERE, 'baseline.json')

from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor  # noqa: E402


def _linearly_separable():
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    return X + 2 * np.random.RandomState(2).uniform(size=X.shape), y


def _titanic_like():
    from test_classifier import titanic_like
    X, y = titanic_like(n=250, seed=1)
    X['age'] = X['age'].fillna(X['age'].median())  # tree NaN support differs across scikit-learn versions
    return X, y


def _first(n, loader):
    """First n rows of a shuffled dataset: keeps the suite within a CI budget (PDL trains on n² pairs)."""
    X, y = loader()
    order = np.random.RandomState(0).permutation(len(y))[:n]
    return X[order], np.asarray(y)[order]


def _small_class():
    X, y = load_wine(return_X_y=True)
    keep = (y != 2) | (np.cumsum(y == 2) <= 8)  # 8 rows in class 2: some folds use the prior path
    return X[keep], y[keep]


DATASETS = {
    'iris': lambda: load_iris(return_X_y=True),
    'wine': lambda: load_wine(return_X_y=True),
    'breast_cancer_250': lambda: _first(250, lambda: load_breast_cancer(return_X_y=True)),
    'digits_300': lambda: _first(300, lambda: load_digits(return_X_y=True)),
    'moons': lambda: make_moons(n_samples=200, noise=0.3, random_state=0),
    'circles': lambda: make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
    'linearly_separable': _linearly_separable,
    'synthetic_3class': lambda: make_classification(n_samples=200, n_features=12, n_informative=6, n_classes=3, random_state=3),
    'titanic_like': _titanic_like,
    'wine_small_class': _small_class,
}

def _house_like(n=200, seed=0):
    """Mixed-dtype regression table: a numeric size, a categorical district, a bool and a noisy price."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    district = pd.Categorical(rng.choice(['north', 'south', 'east', 'west'], size=n))
    size = rng.uniform(30, 150, size=n)
    garden = rng.rand(n) < 0.4
    price = size * np.array([3., 2., 2.5, 1.5])[district.codes] + 40 * garden + rng.normal(scale=20, size=n)
    return pd.DataFrame({'size': size, 'district': district, 'garden': garden, 'age': rng.randint(0, 80, size=n)}), price


REGRESSION_DATASETS = {
    'diabetes_200': lambda: _first(200, lambda: load_diabetes(return_X_y=True)),
    'friedman1': lambda: make_friedman1(n_samples=200, noise=1.0, random_state=0),
    'friedman2': lambda: make_friedman2(n_samples=200, noise=50.0, random_state=0),
    'friedman3': lambda: make_friedman3(n_samples=200, noise=0.1, random_state=0),
    'linear_sparse': lambda: make_regression(n_samples=200, n_features=15, n_informative=5, noise=10.0, random_state=0),
    'house_like': _house_like,
}

BASE_REGRESSORS = {
    'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=50, random_state=53, n_jobs=-1),
    'RandomForest': lambda: RandomForestRegressor(n_estimators=50, random_state=53, n_jobs=-1),
    'DecisionTree': lambda: DecisionTreeRegressor(random_state=53),
}

BASE_LEARNERS = {
    'ExtraTrees': lambda **kw: ExtraTreesClassifier(n_estimators=50, random_state=53, n_jobs=-1, **kw),
    'RandomForest': lambda **kw: RandomForestClassifier(n_estimators=50, random_state=53, n_jobs=-1, **kw),
    'DecisionTree': lambda **kw: DecisionTreeClassifier(random_state=53, **kw),
}


def _encode_for_base(X):
    """The bare base learners need numbers; PDC gets the raw DataFrame."""
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        encoded = {c: X[c].to_numpy(float) if pd.api.types.is_numeric_dtype(X[c]) else
                   (X[c].cat.codes.to_numpy(float) if isinstance(X[c].dtype, pd.CategoricalDtype) else pd.factorize(X[c])[0].astype(float))
                   for c in X.columns}
        return pd.DataFrame(encoded).to_numpy()
    return X


def _cross_validate(make_model, X, y, encode=False, regression=False):
    import pandas as pd
    scores, seconds = [], 0.
    folds = (KFold if regression else StratifiedKFold)(n_splits=5, shuffle=True, random_state=0)
    for train, test in folds.split(np.zeros(len(y)), y):
        X_input = _encode_for_base(X) if encode else X
        take = (lambda idx: X_input.iloc[idx]) if isinstance(X_input, pd.DataFrame) else (lambda idx: X_input[idx])
        start = time.perf_counter()
        model = make_model().fit(take(train), y[train])
        prediction = model.predict(take(test))
        seconds += time.perf_counter() - start
        scores.append(r2_score(y[test], prediction) if regression else f1_score(y[test], prediction, average='macro'))
    return float(np.mean(scores)), seconds


def run(datasets=None, learners=None, verbose=True):
    results = {}
    for data_name in datasets or DATASETS:
        X, y = DATASETS[data_name]()
        y = np.asarray(y)
        results[data_name] = {}
        for learner_name in learners or BASE_LEARNERS:
            make_base = BASE_LEARNERS[learner_name]
            base_f1, base_s = _cross_validate(make_base, X, y, encode=True)
            pdc_f1, pdc_s = _cross_validate(lambda: PairwiseDifferenceClassifier(make_base(class_weight='balanced')), X, y)
            results[data_name][learner_name] = {'base_f1': round(base_f1, 4), 'pdc_f1': round(pdc_f1, 4),
                                                'base_seconds': round(base_s, 3), 'pdc_seconds': round(pdc_s, 3)}
            if verbose:
                print(f'{data_name:20} {learner_name:13} base F1 {base_f1:.4f}  PDC F1 {pdc_f1:.4f}  '
                      f'time base {base_s:6.2f}s  PDC {pdc_s:6.2f}s', flush=True)
    return results


def run_regression(datasets=None, learners=None, verbose=True):
    results = {}
    for data_name in datasets or REGRESSION_DATASETS:
        X, y = REGRESSION_DATASETS[data_name]()
        y = np.asarray(y, dtype=float)
        results[data_name] = {}
        for learner_name in learners or BASE_REGRESSORS:
            make_base = BASE_REGRESSORS[learner_name]
            base_r2, base_s = _cross_validate(make_base, X, y, encode=True, regression=True)
            pdr_r2, pdr_s = _cross_validate(lambda: PairwiseDifferenceRegressor(make_base()), X, y, regression=True)
            results[data_name][learner_name] = {'base_r2': round(base_r2, 4), 'pdr_r2': round(pdr_r2, 4),
                                                'base_seconds': round(base_s, 3), 'pdr_seconds': round(pdr_s, 3)}
            if verbose:
                print(f'{data_name:20} {learner_name:13} base R2 {base_r2:.4f}  PDR R2 {pdr_r2:.4f}  '
                      f'time base {base_s:6.2f}s  PDR {pdr_s:6.2f}s', flush=True)
    return results


def summary(results, model='pdc', metric='f1'):
    rows = [r for per_data in results.values() for r in per_data.values()]
    return {
        f'mean_base_{metric}': float(np.mean([r[f'base_{metric}'] for r in rows])),
        f'mean_{model}_{metric}': float(np.mean([r[f'{model}_{metric}'] for r in rows])),
        f'{model}_wins': int(sum(r[f'{model}_{metric}'] > r[f'base_{metric}'] for r in rows)),
        f'{model}_losses': int(sum(r[f'{model}_{metric}'] < r[f'base_{metric}'] for r in rows)),
        f'{model}_seconds': float(sum(r[f'{model}_seconds'] for r in rows)),
        'base_seconds': float(sum(r['base_seconds'] for r in rows)),
    }


def regression_summary(results):
    return summary(results, model='pdr', metric='r2')


if __name__ == '__main__':
    print(json.dumps(summary(run()), indent=1))
    print(json.dumps(regression_summary(run_regression()), indent=1))
