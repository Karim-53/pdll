"""Test helpers shared by the unit, golden and reference-math tests."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


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
