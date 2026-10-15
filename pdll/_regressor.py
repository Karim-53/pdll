"""Pairwise Difference Regressor (PDR) and its anchor-weighting methods."""
import functools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

from ._base import PairwiseDifferenceMixin, all_pairs, iter_test_chunks, pair_features


class PairwiseDifferenceRegressor(RegressorMixin, PairwiseDifferenceMixin, BaseEstimator):
    """Pairwise Difference Regressor (PDR), a scikit-learn meta-estimator.

    The base estimator learns the target *difference* ``y_a - y_b`` from the pair ``[x_a, x_b, x_a - x_b]``.
    A new point is predicted as ``y_anchor + predicted difference``, averaged over the training points
    (anchors), with the difference symmetrised over both pair orders. PDR is the version by Belaid et al. 2024
    of the PAirwise Difference REgressor (PADRE) of Tynes et al. 2021.

    After fitting, :meth:`learn_anchor_weights` can learn a weight per anchor on validation data.

    Parameters
    ----------
    estimator : regressor, default=None
        Regressor used on the pairs. It is cloned, never modified. ``None`` means ``ExtraTreesRegressor(n_jobs=-1)``.

    max_anchors : int or None, default=500
        If the training set is larger, a random subset of ``max_anchors`` rows is used as anchors.
        ``None`` uses every training row.

    max_pairs : "auto", int or None, default="auto"
        If n² exceeds ``max_pairs``, the base estimator is trained on a uniform random subset of pairs. ``"auto"``
        means 100,000 pairs (every pair up to 316 training rows), fewer for very wide data. ``None`` uses every pair.
        Regression trees grow one leaf per distinct target difference, so with wide data
        ``ExtraTreesRegressor(max_features=0.3)`` trains about 3x faster for a small loss in accuracy.

    random_state : int, RandomState instance or None, default=None
        Seeds the pair and anchor subsampling, and the base estimator if its own ``random_state`` is None.

    Attributes
    ----------
    estimator_ : regressor
        The fitted clone of ``estimator``.
    X_anchors_ : ndarray of shape (n_anchors, n_features)
        Anchors, after encoding of categorical columns.
    y_anchors_ : ndarray of shape (n_anchors,)
        Target of every anchor.
    sample_weight_ : pandas.Series of shape (n_anchors,) or None
        Weight of every anchor, indexed like the training rows; None means equal weights.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Column names seen during fit, if they were all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from pdll import PairwiseDifferenceRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> pdr = PairwiseDifferenceRegressor(DecisionTreeRegressor(random_state=0)).fit(X[:100], y[:100])
    >>> pdr.predict(X[100:102]).shape
    (2,)
    """

    def __init__(self, estimator=None, *, max_anchors=500, max_pairs='auto', random_state=None):
        self.estimator = estimator
        self.max_anchors = max_anchors
        self.max_pairs = max_pairs
        self.random_state = random_state

    def _default_estimator(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_jobs=-1)

    def fit(self, X, y, sample_weight=None):
        """Fit the base estimator on pairs of training points.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data. DataFrame columns of string, object or category dtype are ordinal-encoded.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Row weights. The pair (i, j) gets weight ``w_i * w_j`` and anchors are weighted by ``w``.

        Returns
        -------
        self : object
        """
        max_pairs, max_anchors = self._check_params()
        X, y, sample_weight, row_labels = self._fit_prepare(X, y, sample_weight, y_numeric=True)
        if y.dtype.kind != 'f':
            y = y.astype(np.float64)

        seed_estimator, seed_pairs, seed_anchors = self._random_seeds()
        self.estimator_ = self._make_estimator(seed_estimator, sample_weight)
        left, right = self._sample_pairs(len(X), self._resolve_max_pairs(max_pairs, len(X), X.shape[1]), seed_pairs)
        X_pairs = pair_features(X, X, left, right, order='F')
        y_pairs = y[left] - y[right]
        if sample_weight is None:
            self.estimator_.fit(X_pairs, y_pairs)
        else:
            self.estimator_.fit(X_pairs, y_pairs, sample_weight=sample_weight[left] * sample_weight[right])
        del X_pairs

        anchors = self._sample_anchors(len(X), max_anchors, seed_anchors)
        self.X_anchors_ = X[anchors]
        self.y_anchors_ = y[anchors]
        self._anchor_index = pd.Index(row_labels[anchors])
        self.sample_weight_ = None
        if sample_weight is not None:
            self.sample_weight_ = pd.Series(sample_weight[anchors] / sample_weight[anchors].sum(), index=self._anchor_index)
        return self

    def _predict_samples(self, X, force_symmetry=True):
        """Prediction given each anchor separately.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_anchors)
            ``y_anchor + predicted difference`` for every (sample, anchor) pair.
        differences : ndarray of shape (n_samples, n_anchors)
            The predicted differences alone.
        """
        X = self._validate_X_predict(X)
        anchors = self.X_anchors_
        differences = np.empty((X.shape[0], anchors.shape[0]))
        for rows in iter_test_chunks(X.shape[0], anchors.shape[0], X.shape[1]):
            n_rows = rows.stop - rows.start
            test, anchor = all_pairs(n_rows, anchors.shape[0])
            difference = np.asarray(self.estimator_.predict(pair_features(X[rows], anchors, test, anchor)), dtype=np.float64)
            if force_symmetry:
                difference_sym = np.asarray(self.estimator_.predict(pair_features(anchors, X[rows], anchor, test)), dtype=np.float64)
                difference = (difference - difference_sym) / 2.
            differences[rows] = difference.reshape(n_rows, anchors.shape[0])
        return self.y_anchors_[np.newaxis, :] + differences, differences

    def predict(self, X, force_symmetry=True):
        """Predict the target as the (weighted) mean over anchors of ``y_anchor + predicted difference``.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        force_symmetry : bool, default=True
            Average the predicted difference over both pair orders.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        """
        samples, _ = self._predict_samples(X, force_symmetry=force_symmetry)
        if self.sample_weight_ is None:
            return samples.mean(axis=1)
        weights = np.clip(self.sample_weight_.to_numpy(dtype=np.float64), 0., None)  # non-positive weight = ignored anchor
        if weights.sum() <= 0:
            return samples.mean(axis=1)
        return samples @ weights / weights.sum()

    # ------------------------------------------------------------------ anchor weighting

    @property
    def _name_to_method_mapping(self):
        return {
            # Optimization based methods:
            'L2': functools.partial(self._sample_weight_optimize, l2_lambda=0.1),  # Recommended method
            'KLD': functools.partial(self._sample_weight_optimize, kld_lambda=0.05),
            'Optimize': self._sample_weight_optimize,
            'L1L2': functools.partial(self._sample_weight_optimize, l1_lambda=0.05, l2_lambda=0.025),
            'L1': functools.partial(self._sample_weight_optimize, l1_lambda=0.1),
            'ExtremeWeightPruning': self._sample_weight_extreme_pruning,
            # Heuristic methods
            'NegativeError': self._sample_weight_negative_error,
            'InverseError': self._sample_weight_inverse_error,
            'OrderedVoting': self._sample_weight_ordered_votes,
            # Other Methods:
            'KMeansClusterCenters': self._sample_weight_by_kmeans_prototypes,
        }

    def learn_anchor_weights(self, X_val=None, y_val=None, X_test=None, method='L2', enable_warnings=True, **kwargs):
        """Learn a weight per anchor on validation data (call after :meth:`fit`).

        Methods:

        - ``'Optimize'``: minimize the validation MAE (SLSQP) with the weights summing to 1.
        - ``'L1'``, ``'L2'``, ``'L1L2'``: like ``'Optimize'`` with L1 and/or L2 regularization (``'L2'`` is recommended).
        - ``'KLD'``: like ``'Optimize'`` with a KL-divergence penalty towards uniform weights.
        - ``'ExtremeWeightPruning'``: like ``'L1'`` with a high regularization strength.
        - ``'NegativeError'``, ``'InverseError'``: weights from the validation MAE of each anchor.
        - ``'OrderedVoting'``: the best of n anchors gets n votes, the worst gets 1.
        - ``'KMeansClusterCenters'``: keep only the anchors closest to KMeans centroids (``k``, default 10% of anchors).

        Parameters
        ----------
        X_val, y_val : validation data (not needed by ``'KMeansClusterCenters'``).
        X_test : ignored, kept for backward compatibility.
        method : str, default='L2'
        enable_warnings : bool, default=True
            Warn if the validation MAE got worse.
        **kwargs : extra parameters of the method (e.g. ``k``).

        Returns
        -------
        self : object
        """
        check_is_fitted(self)
        methods = self._name_to_method_mapping
        if method not in methods:
            raise ValueError(f"Weighting method {method} unknown! Use one of the following: '{', '.join(methods)}'")
        old_validation_error = mean_absolute_error(y_val, self.predict(X_val)) if y_val is not None else 0.
        sample_weight = methods[method](X_val=X_val, y_val=y_val, X_test=X_test, **kwargs)
        self.set_sample_weight(sample_weight)
        if y_val is not None and enable_warnings:
            new_validation_error = mean_absolute_error(y_val, self.predict(X_val))
            if new_validation_error > old_validation_error:
                warnings.warn(f'Anchor weights increased the validation MAE from {old_validation_error:.6g} to {new_validation_error:.6g}.',
                              UserWarning, stacklevel=2)
        return self

    def set_sample_weight(self, sample_weight):
        """Set the anchor weights (a pandas Series indexed like the anchors, or an array of length n_anchors).

        Returns
        -------
        self : object
        """
        check_is_fitted(self)
        if sample_weight is None:
            self.sample_weight_ = None
            return self
        if not isinstance(sample_weight, pd.Series):
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim != 1:
                raise ValueError(f'sample_weight must be 1d, got shape {sample_weight.shape}.')
            sample_weight = pd.Series(sample_weight, index=self._anchor_index[:len(sample_weight)])
        if len(sample_weight) != len(self.y_anchors_):
            raise ValueError(f'sample_weight size {len(sample_weight)} should be equal to the number of anchors {len(self.y_anchors_)}')
        if not sample_weight.index.equals(self._anchor_index):
            raise ValueError(f'sample_weight and the anchors must have the same index\n{sample_weight.index}\n{self._anchor_index}')
        sample_weight = sample_weight.astype(np.float64)
        if (sample_weight.fillna(0) == 0).all():  # All weights are 0 => equal weights
            sample_weight = pd.Series(1., index=self._anchor_index)
        if (sample_weight.fillna(0) < 0).all():
            raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')
        if sample_weight.isna().any():
            raise ValueError(f'sample_weight contains NaNs.\n{sample_weight}')
        self.sample_weight_ = sample_weight
        return self

    def _as_weights(self, values):
        return pd.Series(np.asarray(values, dtype=np.float64), index=self._anchor_index)

    @staticmethod
    def _normalize_weights_to_0_to_1(weights: pd.Series) -> pd.Series:
        """Normalize non-negative weights to sum to 1 (equal weights if they are all equal)."""
        if np.allclose(weights, weights.values[0]):
            weights = pd.Series(1., index=weights.index)
        if weights.min() < 0:
            raise ValueError(f'Negative weights found: {weights[weights < 0]}')
        return weights / weights.sum()

    @staticmethod
    def _objective_function(weights, pred_val_samples_np, y_val, initial_mae, kld_lambda=0., l1_lambda=0., l2_lambda=0.) -> float:
        if min(kld_lambda, l1_lambda, l2_lambda) < 0 or initial_mae < 0:
            raise ValueError('Regularization strengths and initial_mae must be >= 0.')
        predictions = np.matmul(pred_val_samples_np, weights / sum(weights))
        mae = mean_absolute_error(y_val, predictions)

        regularisation = 0
        if kld_lambda > 0:
            train_size = len(weights)
            weights_initial_guess = np.ones(train_size) / train_size
            regularisation += kld_lambda * entropy(weights, weights_initial_guess) / train_size
        if l1_lambda > 0:
            regularisation += l1_lambda * (np.linalg.norm(weights, ord=1) - max(weights))
        if l2_lambda > 0:
            regularisation += l2_lambda * np.linalg.norm(weights, ord=2)
        return mae + regularisation * initial_mae

    def _sample_weight_optimize(self, X_val, y_val, kld_lambda=0., l1_lambda=0., l2_lambda=0., **kwargs) -> pd.Series:
        """Minimize the validation MAE with SLSQP under the constraint sum(weights) = 1."""
        pred_val_samples_np, _ = self._predict_samples(X_val)
        # Column-major like pandas' DataFrame.values in pdll <= 0.3: the BLAS summation order (and therefore the SLSQP path,
        # which amplifies 1-ulp differences) then matches the published weighted-PDR results exactly.
        pred_val_samples_np = np.asfortranarray(pred_val_samples_np)
        y_val = np.asarray(y_val, dtype=np.float64)
        train_size = pred_val_samples_np.shape[1]
        weights_initial_guess = np.ones(train_size) / train_size
        initial_mae = mean_absolute_error(y_val, np.matmul(pred_val_samples_np, weights_initial_guess))

        def objective_function(weights):
            return self._objective_function(weights, pred_val_samples_np, y_val, initial_mae,
                                            kld_lambda=kld_lambda, l1_lambda=l1_lambda, l2_lambda=l2_lambda)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = minimize(objective_function, weights_initial_guess, method='SLSQP',
                              bounds=[(0., 1.)] * train_size, constraints=[LinearConstraint(np.ones(train_size), lb=1, ub=1)])
        return self._as_weights(result.x)

    def _sample_weight_extreme_pruning(self, X_val, y_val, **kwargs) -> pd.Series:
        l1 = 0.8
        while l1 > 0.0001:
            weights = self._sample_weight_optimize(X_val=X_val, y_val=y_val, l1_lambda=l1)
            if sum(weights == 0) / len(weights) > .9:
                l1 *= 0.5
            else:
                break
        return weights

    def _error(self, X_val, y_val, **kwargs) -> pd.Series:
        """Validation mean absolute error of each anchor."""
        pred_val_samples, _ = self._predict_samples(X_val)
        y_val = np.asarray(y_val, dtype=np.float64)
        return self._as_weights(np.abs(y_val[:, np.newaxis] - pred_val_samples).mean(axis=0))

    def _sample_weight_inverse_error(self, X_val, y_val, **kwargs) -> pd.Series:
        val_mae = self._error(X_val=X_val, y_val=y_val)
        sample_weights = 1. / (val_mae + 0.0001)
        return sample_weights / sample_weights.sum()

    def _sample_weight_negative_error(self, X_val, y_val, **kwargs) -> pd.Series:
        uniform_weights = self._as_weights(np.full(len(self.y_anchors_), 1 / len(self.y_anchors_)))
        val_mae = self._error(X_val=X_val, y_val=y_val)
        if sum(val_mae) == 0:
            return uniform_weights
        sample_weights = ((-val_mae) + max(val_mae)) / sum(val_mae)
        if sum(sample_weights) == 0:
            return uniform_weights
        return sample_weights / sample_weights.sum()

    @staticmethod
    def _sample_weight_ordered_votes_from_weights(received_weights):
        errors = -np.asarray(received_weights, dtype=np.float64)
        k = len(errors)
        ranks = np.argsort(np.argsort(errors)) + 1
        weights = (k - ranks + 1) / (k * (k + 1) / 2)
        if isinstance(received_weights, pd.Series):
            return pd.Series(weights, index=received_weights.index)
        return weights

    def _sample_weight_ordered_votes(self, X_val, y_val, force_symmetry=True, **kwargs):
        """The best of n anchors gets n votes, the worst gets 1 vote (ranked by NegativeError weights)."""
        weights = self._sample_weight_negative_error(X_val, y_val, force_symmetry=force_symmetry)
        return self._sample_weight_ordered_votes_from_weights(weights)

    def _sample_weight_by_kmeans_prototypes(self, k=None, **kwargs):
        """Keep only the k anchors closest to the KMeans centroids (equal weights), all others get 0."""
        from sklearn.cluster import KMeans
        n_anchors = len(self.X_anchors_)
        if not k:
            k = max(int(n_anchors / 10), 3)  # 10% of the anchors, at least 3
        k = min(k, n_anchors)
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(self.X_anchors_)
        closest_indices = np.argmin(cdist(self.X_anchors_, kmeans.cluster_centers_), axis=0)
        closest_array = np.zeros(n_anchors)
        closest_array[closest_indices] = 1 / k
        return self._as_weights(closest_array)
