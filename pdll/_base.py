"""Machinery shared by the PDL classifier and regressor: pairs, subsampling, anchors, input checks."""
import numbers
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from ._compat import NEW_TAGS_API, allows_nan
from ._input import check_sample_weight, to_float_array

# Upper bound on the number of cells (pairs x features) materialised at once when predicting.
CHUNK_CELLS = 2 ** 24

# max_pairs='auto': at most DEFAULT_MAX_PAIRS training pairs (all of them up to 316 rows), and fewer for very wide data so
# that the pair matrix stays below PAIR_CELL_BUDGET cells (0.5 GB in float64). Chosen in the 2026-10 study on real datasets
# (benchmark/real_world): versus 1M pairs, mean macro-F1 -0.005 for ~10x less time and memory, and no out-of-memory failures.
DEFAULT_MAX_PAIRS = 100_000
PAIR_CELL_BUDGET = 2 ** 26

# Above this many training rows, PDL was usually less accurate than its base estimator alone on 130 real datasets
# (benchmark/real_world, 2026-10): it is a small-data method. fit() says so.
LARGE_DATA_ROWS = 2000

# Base learners whose predictions rarely improve when wrapped in PDL (see the paper, section 5).
_PARAMETRIC_HINT = ('PDL is not compatible with base estimators of type parametric models, i.e., even if the code works there is '
                    'low chance of improvement compared to using directly that estimator. To obtain an improvement, it is better '
                    'to use a tree-based model like: ExtraTrees, RandomForest, DecisionTree, Bagging, etc.')


def _parametric_estimators():
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR, LinearSVC
    return (MLPClassifier, KNeighborsClassifier, GaussianNB, BernoulliNB, SGDClassifier, LinearSVC, LogisticRegression,
            RidgeClassifier, GaussianProcessClassifier, SVC, MultinomialNB, LinearRegression, MLPRegressor,
            KNeighborsRegressor, GaussianProcessRegressor, Ridge, Lasso, ElasticNet, SVR)


def pair_features(X_left, X_right, left, right, order='C'):
    """Rows ``[X_left[left], X_right[right], X_left[left] - X_right[right]]``, the PDL pair representation.

    Use ``order='F'`` for training: tree splitters scan one feature across samples, which is faster column-major.
    """
    n_features = X_left.shape[1]
    pairs = np.empty((len(left), 3 * n_features), dtype=np.result_type(X_left, X_right), order=order)
    pairs[:, :n_features] = X_left[left]
    pairs[:, n_features:2 * n_features] = X_right[right]
    np.subtract(pairs[:, :n_features], pairs[:, n_features:2 * n_features], out=pairs[:, 2 * n_features:])
    return pairs


def iter_test_chunks(n_test, n_anchors, n_features):
    """Slices of test rows such that one chunk of pairs stays below CHUNK_CELLS cells."""
    rows = max(1, CHUNK_CELLS // max(1, n_anchors * 3 * n_features))
    for start in range(0, n_test, rows):
        yield slice(start, min(start + rows, n_test))


def all_pairs(n_test, n_anchors):
    """Indices of every (test, anchor) pair, test-major: the row order PDL has always used."""
    return np.repeat(np.arange(n_test), n_anchors), np.tile(np.arange(n_anchors), n_test)


def _check_max(value, name, allow_auto=False):
    if value is None:
        return None
    if allow_auto and isinstance(value, str) and value == 'auto':
        return 'auto'
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        options = 'None, "auto"' if allow_auto else 'None'
        raise ValueError(f'{name} must be {options} or a positive integer, got {value!r}.')
    return int(value)


class PairwiseDifferenceMixin:
    """Fit/predict helpers shared by PairwiseDifferenceClassifier and PairwiseDifferenceRegressor."""

    def _default_estimator(self):
        raise NotImplementedError

    def _base_estimator(self):
        return self._default_estimator() if self.estimator is None else self.estimator

    def _allow_nan(self):
        return allows_nan(self._base_estimator())

    if NEW_TAGS_API:  # scikit-learn >= 1.6 (defining _more_tags there triggers deprecation warnings)
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = self._allow_nan()
            return tags
    else:  # pragma: no cover - scikit-learn < 1.6
        def _more_tags(self):
            return {'allow_nan': self._allow_nan()}

    def _validate_X(self, X, y='no_validation', reset=False, **check_params):
        return to_float_array(self, X, y, reset=reset, allow_nan=self._allow_nan(), **check_params)

    def _validate_X_predict(self, X):
        check_is_fitted(self)
        return self._validate_X(X, reset=False)

    def _check_params(self):
        estimator = self._base_estimator()
        if isinstance(estimator, type):
            raise TypeError('estimator must be an instance of the class not a class, i.e., use MyEstimator() but not MyEstimator')
        if not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
            raise TypeError(f'estimator must implement fit and predict, got {estimator!r}.')
        return _check_max(self.max_pairs, 'max_pairs', allow_auto=True), _check_max(self.max_anchors, 'max_anchors')

    @staticmethod
    def _warn_if_large(n_samples):
        if n_samples > LARGE_DATA_ROWS:
            warnings.warn(f'PDL is designed for small datasets. With {n_samples} training rows it uses a sample of the pairs and '
                          f'anchors (max_pairs, max_anchors), and in our benchmark of 130 real datasets it was usually less accurate '
                          f'than its base estimator alone above {LARGE_DATA_ROWS} rows. Compare both on your data, or raise '
                          f'max_pairs and max_anchors.', UserWarning, stacklevel=3)

    @staticmethod
    def _resolve_max_pairs(max_pairs, n_samples, n_features):
        """Turn max_pairs='auto' into a number; warn when memory, not the default, limits very wide data."""
        if max_pairs != 'auto':
            return max_pairs
        memory_cap = max(1, PAIR_CELL_BUDGET // (3 * max(1, n_features)))
        if memory_cap < min(DEFAULT_MAX_PAIRS, n_samples * n_samples) and memory_cap < 10 * n_samples:
            warnings.warn(f'With {n_features} features, only {memory_cap} training pairs fit in the memory budget of '
                          f'max_pairs="auto" ({n_samples} training rows). PDL is designed for tabular data with fewer features: '
                          f'consider feature selection, or set max_pairs explicitly.', UserWarning, stacklevel=4)
        return min(DEFAULT_MAX_PAIRS, memory_cap)

    def _random_seeds(self):
        """Independent seeds for (base estimator, pair sampling, anchor sampling), all derived from random_state."""
        random_state = check_random_state(self.random_state)
        return [int(seed) for seed in random_state.randint(np.iinfo(np.int32).max, size=3)]

    def _make_estimator(self, seed, sample_weight):
        estimator = clone(self._base_estimator(), safe=False)
        if self.random_state is not None and hasattr(estimator, 'get_params'):
            if 'random_state' in estimator.get_params(deep=False) and estimator.get_params(deep=False)['random_state'] is None:
                estimator.set_params(random_state=seed)
        if sample_weight is not None and not has_fit_parameter(estimator, 'sample_weight'):
            raise ValueError(f'{type(estimator).__name__} does not support sample_weight, so PDL cannot use it either.')
        if isinstance(estimator, _parametric_estimators()):
            warnings.warn(_PARAMETRIC_HINT, UserWarning, stacklevel=3)
        return estimator

    @staticmethod
    def _sample_pairs(n_samples, max_pairs, seed):
        """Indices (left, right) of the training pairs: all n² ordered pairs, or a uniform subset of max_pairs of them."""
        if max_pairs is None or n_samples * n_samples <= max_pairs:
            return all_pairs(n_samples, n_samples)
        flat = np.random.default_rng(seed).choice(n_samples * n_samples, size=max_pairs, replace=False)
        flat.sort()  # keep the test-major row order
        return flat // n_samples, flat % n_samples

    @staticmethod
    def _sample_anchors(n_samples, max_anchors, seed, groups=None):
        """Sorted indices of the training rows kept as anchors (stratified by ``groups`` when given)."""
        if max_anchors is None or n_samples <= max_anchors:
            return np.arange(n_samples)
        rng = np.random.default_rng(seed)
        if groups is None:
            return np.sort(rng.choice(n_samples, size=max_anchors, replace=False))
        labels, counts = np.unique(groups, return_counts=True)
        quota = np.maximum(1, np.floor(counts * max_anchors / n_samples).astype(int))
        remainder = max_anchors - quota.sum()
        if remainder > 0:  # largest remainder method
            quota[np.argsort(-(counts * max_anchors / n_samples - quota))[:remainder]] += 1
        quota = np.minimum(quota, counts)
        chosen = [rng.choice(np.flatnonzero(groups == label), size=k, replace=False) for label, k in zip(labels, quota)]
        return np.sort(np.concatenate(chosen))

    def _fit_prepare(self, X, y, sample_weight, **check_params):
        """Validate the fit input and drop zero-weight rows. Returns (X, y, sample_weight or None, row_labels)."""
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
        X, y = self._validate_X(X, y, reset=True, **check_params)
        sample_weight = check_sample_weight(sample_weight, X.shape[0])
        row_labels = np.asarray(index) if index is not None and len(index) == X.shape[0] else np.arange(X.shape[0])
        if sample_weight is not None and np.any(sample_weight == 0):
            keep = sample_weight > 0
            X, y, sample_weight, row_labels = X[keep], y[keep], sample_weight[keep], row_labels[keep]
        return X, y, sample_weight, row_labels
