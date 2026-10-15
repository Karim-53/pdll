"""Datasets and base estimators of the golden (0.3.1 equivalence) tests.

Every case uses numpy inputs and a seeded base estimator, like the paper benchmark.
Changing anything here invalidates ``v0_3_1.npz``.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, make_classification
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from helpers import FakePairClassifier, FakePairRegressor

# Tree based estimators: the outputs depend on the scikit-learn version that built the trees.
SKLEARN_DEPENDENT = ('extratrees', 'randomforest', 'decisiontree', 'histgb', 'dtr', 'etr')


def _split(X, y, n_train, seed):
    order = np.random.RandomState(seed).permutation(len(X))
    train, test = order[:n_train], order[n_train:]
    return X[train], y[train], X[test]


def _datasets():
    X, y = load_iris(return_X_y=True)
    yield 'iris64', _split(X, y, 100, 0)
    yield 'iris32', _split(X.astype(np.float32), y.astype(np.float32), 100, 1)  # benchmark style: float32 X and y codes

    X, y = load_wine(return_X_y=True)
    X_train, y_train, X_test = _split(X, y, 120, 2)
    keep = (y_train != 2) | (np.cumsum(y_train == 2) <= 3)  # class 2 keeps 3 rows: prior path
    yield 'wine_prior', (X_train[keep], y_train[keep], X_test)

    X, y = load_breast_cancer(return_X_y=True)
    yield 'cancer32', _split(X[:220].astype(np.float32), y[:220].astype(np.float32), 140, 3)

    X, y = make_classification(n_samples=260, n_features=6, n_informative=4, n_classes=3, random_state=4)
    yield 'synth3', _split(X, y, 180, 5)


def _classifiers():
    yield 'fake', lambda: FakePairClassifier()
    yield 'extratrees', lambda: ExtraTreesClassifier(n_estimators=30, class_weight='balanced', random_state=53, n_jobs=1)
    yield 'randomforest', lambda: RandomForestClassifier(n_estimators=30, class_weight='balanced', random_state=53, n_jobs=1)
    yield 'decisiontree', lambda: DecisionTreeClassifier(class_weight='balanced', random_state=53)
    yield 'histgb', lambda: HistGradientBoostingClassifier(max_iter=20, class_weight='balanced', random_state=53)


def classification_cases():
    for data_name, data in _datasets():
        for est_name, make in _classifiers():
            if est_name == 'histgb' and data_name not in ('iris64', 'wine_prior'):
                continue  # keep the suite fast
            yield f'{data_name}__{est_name}', data, make


def regression_cases():
    X, y = load_diabetes(return_X_y=True)
    order = np.random.RandomState(6).permutation(len(X))
    X, y = X[order], y[order]
    data = (X[:90], y[:90], X[150:200], X[90:150], y[90:150])
    yield 'diabetes__fake', data, lambda: FakePairRegressor(), None
    yield 'diabetes__dtr', data, lambda: DecisionTreeRegressor(random_state=0), 'OrderedVoting'
    yield 'diabetes__etr', data, lambda: ExtraTreesRegressor(n_estimators=20, random_state=0, n_jobs=1), 'L2'
    yield 'diabetes__fakeL2', data, lambda: FakePairRegressor(), 'L2'
