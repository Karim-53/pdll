"""Pairwise Difference Classifier (PDC)."""
import inspect
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import has_fit_parameter

from ._base import PairwiseDifferenceMixin, all_pairs, iter_test_chunks, pair_features

# With fewer rows than this in some class, use_prior='auto' switches to the prior-based aggregation.
MIN_CLASS_SIZE_WITHOUT_PRIOR = 5


class PairwiseDifferenceClassifier(ClassifierMixin, PairwiseDifferenceMixin, BaseEstimator):
    """Pairwise Difference Classifier (PDC), a scikit-learn meta-estimator.

    PDC turns a (multiclass) classification task into a binary one: the base estimator learns whether
    two points belong to the *same* class from the pair ``[x_a, x_b, x_a - x_b]``. To classify a new point,
    PDC compares it with every training point (the *anchors*), whose labels are known, and turns the
    predicted similarities into class probabilities. Tree ensembles such as ExtraTrees or RandomForest
    typically gain accuracy from this, especially on small tabular datasets.

    Read more in Belaid, Rabus & Hüllermeier, *Pairwise Difference Learning for Classification*,
    Discovery Science 2024, https://arxiv.org/abs/2406.20031 (``pdll.__citation__`` holds the BibTeX).

    Parameters
    ----------
    estimator : classifier, default=None
        Binary classifier used on the pairs. It is cloned, never modified. ``None`` means
        ``ExtraTreesClassifier(class_weight="balanced", n_jobs=-1)``. Prefer tree-based models and
        ``class_weight="balanced"``: only about 1/n_classes of the pairs are "same class".

    use_prior : {"auto", True, False}, default="auto"
        How similarities become probabilities. ``True``: the prior-based posterior of the paper,
        averaged over anchors. ``False``: the mean similarity to each class, normalised.
        ``"auto"``: ``True`` if some class has fewer than 5 training rows, else ``False``.

    class_weight : dict, "balanced" or None, default=None
        Weights of the classes, applied as sample weights (see ``fit``). ``"balanced"`` gives every class the same
        total weight; on imbalanced data this counters PDL's tendency to under-predict minority classes
        (in our benchmark: up to +0.05 macro-F1, never worse). Needs a base estimator whose ``fit`` accepts
        ``sample_weight``. ``None`` (default) weighs every row equally, as in the paper.

    max_anchors : int or None, default=500
        Prediction compares each point with every anchor, so its cost grows with the number of anchors.
        If the training set is larger, a class-stratified random subset of ``max_anchors`` rows is used
        as anchors. ``None`` uses every training row.

    max_pairs : "auto", int or None, default="auto"
        The base estimator is trained on n² pairs. If that exceeds ``max_pairs``, a uniform random subset
        of ``max_pairs`` pairs is used. ``"auto"`` means 100,000 pairs (every pair up to 316 training rows), fewer for
        very wide data so the pair matrix stays below 0.5 GB. ``None`` uses every pair, as in the paper; time and
        memory then grow with n².

    random_state : int, RandomState instance or None, default=None
        Seeds the pair and anchor subsampling, and the base estimator if its own ``random_state`` is None.

    Attributes
    ----------
    estimator_ : classifier
        The fitted clone of ``estimator``.
    classes_ : ndarray of shape (n_classes,)
        The class labels.
    n_classes_ : int
        Number of classes.
    prior_ : ndarray of shape (n_classes,)
        Class frequencies of the training data (weighted by ``sample_weight``).
    X_anchors_ : ndarray of shape (n_anchors, n_features)
        Anchors, after encoding of categorical columns.
    y_anchors_ : ndarray of shape (n_anchors,)
        Class index (into ``classes_``) of every anchor.
    sample_weight_ : ndarray of shape (n_anchors,) or None
        Weight of every anchor at prediction time, or None for equal weights.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Column names seen during fit, if they were all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from pdll import PairwiseDifferenceClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> pdc = PairwiseDifferenceClassifier(RandomForestClassifier(class_weight="balanced", random_state=0))
    >>> pdc.fit(X, y).predict(X[:3])
    array([0, 0, 0])
    """

    def __init__(self, estimator=None, *, use_prior='auto', class_weight=None, max_anchors=500, max_pairs='auto', random_state=None):
        self.estimator = estimator
        self.use_prior = use_prior
        self.class_weight = class_weight
        self.max_anchors = max_anchors
        self.max_pairs = max_pairs
        self.random_state = random_state

    def _default_estimator(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)

    def fit(self, X, y, sample_weight=None):
        """Fit the base estimator on pairs of training points.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data. DataFrame columns of string, object or category dtype are ordinal-encoded.
        y : array-like of shape (n_samples,)
            Class labels (any hashable type).
        sample_weight : array-like of shape (n_samples,), default=None
            Row weights. The pair (i, j) gets weight ``w_i * w_j``, which is equivalent to repeating rows.

        Returns
        -------
        self : object
        """
        if self.use_prior not in ('auto', True, False):
            raise ValueError(f'use_prior must be "auto", True or False, got {self.use_prior!r}.')
        max_pairs, max_anchors = self._check_params()
        X, y, sample_weight, _ = self._fit_prepare(X, y, sample_weight)
        check_classification_targets(y)
        if self.class_weight is not None:
            if not has_fit_parameter(self._base_estimator(), 'sample_weight'):
                raise ValueError(f'class_weight is applied through sample weights, but {type(self._base_estimator()).__name__}.fit '
                                 f'does not accept sample_weight.')
            class_sample_weight = compute_sample_weight(self.class_weight, y)
            sample_weight = class_sample_weight if sample_weight is None else sample_weight * class_sample_weight
            if np.any(sample_weight == 0):  # a class weighted 0 is left out, like a row weighted 0
                keep = sample_weight > 0
                X, y, sample_weight = X[keep], y[keep], sample_weight[keep]
        self._warn_if_large(len(X))
        classes, y_all = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        present = np.unique(y_all)
        if len(present) < 2:
            raise ValueError(f'PairwiseDifferenceClassifier needs at least 2 classes, but only one class ({classes[present[0]]!r}) '
                             f'is present in the training data (among rows with a positive sample_weight).')

        weights = np.ones(len(y_all)) if sample_weight is None else sample_weight
        self.prior_ = np.bincount(y_all, weights=weights, minlength=self.n_classes_) / weights.sum()
        self._min_class_count = int(np.bincount(y_all)[present].min())

        seed_estimator, seed_pairs, seed_anchors = self._random_seeds()
        self.estimator_ = self._make_estimator(seed_estimator, sample_weight)
        self._warn_class_weight()
        left, right = self._sample_pairs(len(X), self._resolve_max_pairs(max_pairs, len(X), X.shape[1]), seed_pairs)
        left, right = self._ensure_both_targets(y_all, left, right)
        X_pairs = pair_features(X, X, left, right, order='F')
        y_pairs = (y_all[left] != y_all[right]).astype(int)  # 0 = same class, 1 = different
        if sample_weight is None:
            self.estimator_.fit(X_pairs, y_pairs)
        else:
            self._fit_weighted_pairs(X_pairs, y_pairs, sample_weight[left] * sample_weight[right])
        del X_pairs

        anchors = self._sample_anchors(len(X), max_anchors, seed_anchors, groups=y_all)
        self.X_anchors_ = X[anchors]
        self.y_anchors_ = y_all[anchors]
        self.sample_weight_ = None if sample_weight is None else sample_weight[anchors] / sample_weight[anchors].sum()
        return self

    def _fit_weighted_pairs(self, X_pairs, y_pairs, pair_weight):
        """Fit the base estimator on weighted pairs, with class_weight="balanced" computed from the weighted counts.

        Before scikit-learn 1.8, forests and trees balance on row counts and ignore sample_weight, so the same weights
        gave a different model on older versions. Folding the balance into the pair weights fits the same model on all.
        """
        if getattr(self.estimator_, 'class_weight', None) != 'balanced':
            self.estimator_.fit(X_pairs, y_pairs, sample_weight=pair_weight)
            return
        weighted_counts = np.bincount(y_pairs, weights=pair_weight, minlength=2)
        balance = weighted_counts.sum() / (len(weighted_counts) * weighted_counts)
        self.estimator_.set_params(class_weight=None)
        self.estimator_.fit(X_pairs, y_pairs, sample_weight=pair_weight * balance[y_pairs])
        self.estimator_.set_params(class_weight='balanced')  # keep the user's setting visible; prediction ignores it

    def _warn_class_weight(self):
        params = inspect.signature(type(self.estimator_).__init__).parameters
        if 'class_weight' in params and getattr(self.estimator_, 'class_weight', None) != 'balanced':
            warnings.warn('For better performance, estimator should have class_weight="balanced".', UserWarning, stacklevel=3)

    @staticmethod
    def _ensure_both_targets(y, left, right):
        """Subsampled pairs could all be "same" or all "different"; add one pair of the missing kind."""
        same = y[left] == y[right]
        if same.all():
            first, other = 0, np.flatnonzero(y != y[0])[0]
        elif not same.any():
            first = other = 0
        else:
            return left, right
        return np.append(left, first), np.append(right, other)

    def decide_use_prior(self) -> bool:
        """Whether predict_proba uses the prior-based aggregation (see ``use_prior``)."""
        if not isinstance(self.use_prior, str):  # True / False, including numpy booleans
            return bool(self.use_prior)
        return self._min_class_count < MIN_CLASS_SIZE_WITHOUT_PRIOR

    def _p_same(self, X_pairs):
        """Probability that each pair is of the same class, according to the base estimator."""
        if hasattr(self.estimator_, 'predict_proba'):
            proba = self.estimator_.predict_proba(X_pairs)
            classes = getattr(self.estimator_, 'classes_', None)
            return proba[:, 0 if classes is None else np.flatnonzero(np.asarray(classes) == 0)[0]]
        return (np.asarray(self.estimator_.predict(X_pairs)) == 0).astype(np.float64)

    def _similarity(self, X):
        """(n_samples, n_anchors) symmetrised probability that X[i] and anchor j are of the same class."""
        anchors = self.X_anchors_
        similarity = np.empty((X.shape[0], anchors.shape[0]))
        for rows in iter_test_chunks(X.shape[0], anchors.shape[0], X.shape[1]):
            n_rows = rows.stop - rows.start
            test, anchor = all_pairs(n_rows, anchors.shape[0])
            p_ab = self._p_same(pair_features(X[rows], anchors, test, anchor))
            p_ba = self._p_same(pair_features(anchors, X[rows], anchor, test))
            similarity[rows] = ((p_ab + p_ba) / 2.).reshape(n_rows, anchors.shape[0])
        return similarity

    def _anchor_weights(self):
        n_anchors = len(self.y_anchors_)
        return np.full(n_anchors, 1. / n_anchors) if self.sample_weight_ is None else self.sample_weight_

    def _posterior_terms(self, similarity):
        """Per anchor j, the paper's posterior is p_c * (1 - s_j) / (1 - p_{y_j}) for c != y_j, and s_j for c = y_j."""
        prior_of_anchor = self.prior_[self.y_anchors_]
        return (1. - similarity) / (1. - prior_of_anchor)[np.newaxis, :]

    def _proba_from_similarity(self, similarity):
        weights = self._anchor_weights()
        one_hot = np.zeros((len(self.y_anchors_), self.n_classes_))
        one_hot[np.arange(len(self.y_anchors_)), self.y_anchors_] = 1.
        if self.decide_use_prior():
            spread = self._posterior_terms(similarity) * weights
            proba = spread.sum(axis=1)[:, np.newaxis] * self.prior_
            proba += (similarity * weights) @ one_hot - (spread @ one_hot) * self.prior_
            proba = proba / proba.sum(axis=1)[:, np.newaxis]
            return proba.clip(0., 1.)
        class_weight = weights @ one_hot
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_similarity = np.where(class_weight > 0, ((similarity * weights) @ one_hot) / class_weight, 0.)
            total = mean_similarity.sum(axis=1)[:, np.newaxis]
            proba = mean_similarity / total
        no_signal = total[:, 0] <= 0  # every similarity is 0: fall back on the prior
        proba[no_signal] = self.prior_
        return proba

    def predict_proba(self, X):
        """Class probabilities.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Columns follow ``classes_``.
        """
        X = self._validate_X_predict(X)
        return self._proba_from_similarity(self._similarity(X))

    def predict_log_proba(self, X):
        """Logarithm of :meth:`predict_proba`."""
        with np.errstate(divide='ignore'):
            return np.log(self.predict_proba(X))

    def predict(self, X):
        """Most probable class.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Labels from ``classes_``.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_similarity_samples(self, X):
        """Symmetrised probability that each sample is of the same class as each anchor.

        Returns
        -------
        similarity : ndarray of shape (n_samples, n_anchors)
        """
        return self._similarity(self._validate_X_predict(X))

    def predict_proba_samples(self, X):
        """Class posterior given each anchor separately (the paper's per-anchor posterior, before averaging).

        Useful for uncertainty quantification: the spread across anchors measures disagreement.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_anchors, n_classes)
        """
        similarity = self.predict_similarity_samples(X)
        posterior = self._posterior_terms(similarity)[:, :, np.newaxis] * self.prior_
        n_samples, n_anchors = similarity.shape
        posterior[np.arange(n_samples)[:, None], np.arange(n_anchors)[None, :], self.y_anchors_[None, :]] = similarity
        return posterior

    def score_difference(self, X, y) -> float:
        """Mean absolute error of the predicted pair dissimilarity against the true one, over all (X, anchor) pairs."""
        similarity = self.predict_similarity_samples(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != similarity.shape[0]:
            raise ValueError(f'y must be 1d with {similarity.shape[0]} labels, got shape {y.shape}.')
        different = (y[:, np.newaxis] != self.classes_[self.y_anchors_][np.newaxis, :]).astype(float)
        return float(np.abs(different - (1. - similarity)).mean())
