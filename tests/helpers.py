"""Test helpers shared by the unit, golden and reference-math tests."""
import json
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def check_score(score, reference, ratio=0.95):
    """Fail when a held-out score drops more than ``1 - ratio`` of ``|reference|`` below the value it had when the check
    was written (for a positive reference: below ``ratio`` x ``reference``).

    References were measured with pdll 1.0.0 on the latest stack (scikit-learn 1.9, Windows x86-64). The margin
    absorbs the slightly different trees that other platforms and older scikit-learn versions build; a real
    regression shows up as a drop well beyond it. Raise the reference when a change improves the score.
    Set ``PDLL_SCORE_LOG=<file>`` to record every measured score, e.g. to refresh the references.
    """
    log = os.environ.get('PDLL_SCORE_LOG')
    if log:
        with open(log, 'a') as f:
            f.write(json.dumps({'test': os.environ.get('PYTEST_CURRENT_TEST', '').split(' ')[0],
                                'score': float(score), 'reference': reference}) + '\n')
    if reference is not None:
        floor = reference - (1 - ratio) * abs(reference)
        assert score >= floor, f'score {score:.4f} < floor {floor:.4f} ({ratio} of reference {reference:.4f})'


class FakePairClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic stand-in for the base estimator of PairwiseDifferenceClassifier.

    P(same) is a fixed function of the pair features ``[a, b, a - b]``. It is deliberately
    asymmetric in (a, b) so that the symmetric averaging of PDL is exercised. ``fit`` records
    what it was trained on so tests can check how the pairs were built.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y, sample_weight=None):
        self.X_fit_ = np.array(X, copy=True)
        self.y_fit_ = np.array(y, copy=True)
        self.sample_weight_fit_ = None if sample_weight is None else np.array(sample_weight, copy=True)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1] // 3
        a, b, d = X[:, :n_features], X[:, n_features:2 * n_features], X[:, 2 * n_features:]
        logit = 1.0 - self.scale * np.abs(d).sum(axis=1) / n_features + 0.3 * np.tanh(a[:, 0] - 0.5 * b[:, 0])
        p_same = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([p_same, 1.0 - p_same])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class FakePairRegressor(RegressorMixin, BaseEstimator):
    """Deterministic stand-in for the base estimator of PairwiseDifferenceRegressor."""

    def fit(self, X, y, sample_weight=None):
        self.X_fit_ = np.array(X, copy=True)
        self.y_fit_ = np.array(y, copy=True)
        self.sample_weight_fit_ = None if sample_weight is None else np.array(sample_weight, copy=True)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1] // 3
        d = X[:, 2 * n_features:]
        return d.sum(axis=1) + 0.1 * X[:, 0]
