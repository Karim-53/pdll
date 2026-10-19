"""Unit tests of PairwiseDifferenceClassifier: inputs, labels, maths, subsampling, weights, sklearn tooling."""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pdll._base
from helpers import FakePairClassifier
from pdll import PairwiseDifferenceClassifier


def forest(**kwargs):
    return ExtraTreesClassifier(n_estimators=20, class_weight='balanced', random_state=0, **kwargs)


def pdc(estimator=None, **kwargs):
    return PairwiseDifferenceClassifier(forest() if estimator is None else estimator, **kwargs)


@pytest.fixture(scope='module')
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture(scope='module')
def cancer_frame():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X.iloc[:, :8], y, train_size=160, test_size=80, random_state=0, stratify=y)


@pytest.fixture
def small_classes(iris):
    """Every 15th iris row: 3-4 rows per class, so use_prior='auto' picks the prior path."""
    return iris[0][::15], iris[1][::15]


# --------------------------------------------------------------------------- outputs

def test_predict_returns_ndarray_of_original_labels(cancer_frame):
    X_train, X_test, y_train, _ = cancer_frame
    labels = np.where(y_train == 1, 'benign', 'malignant')
    pred = pdc().fit(X_train, labels).predict(X_test)
    assert isinstance(pred, np.ndarray) and pred.shape == (len(X_test),)
    assert set(pred) <= {'benign', 'malignant'}


def test_predict_proba_is_ndarray_usable_for_roc_auc(cancer_frame):
    X_train, X_test, y_train, y_test = cancer_frame
    proba = pdc().fit(X_train, y_train).predict_proba(X_test)
    assert isinstance(proba, np.ndarray) and proba.shape == (len(X_test), 2)
    assert roc_auc_score(y_test, proba[:, 1]) > 0.9


def test_proba_columns_follow_classes_and_sum_to_one(iris):
    X, y = iris
    model = pdc().fit(X, y + 10)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))
    np.testing.assert_array_equal(model.classes_, [10, 11, 12])
    np.testing.assert_allclose(np.exp(model.predict_log_proba(X)), proba)


# --------------------------------------------------------------------------- labels

@pytest.mark.parametrize('labels', [['a', 'b', 'c'], [0, 5, 9], [-1., 2., 7.], [True, False]])
@pytest.mark.parametrize('path', ['prior', 'mean'])
def test_any_label_values(iris, small_classes, labels, path):
    X, y = small_classes if path == 'prior' else iris
    X, y = X[y < len(labels)], y[y < len(labels)]
    mapped = np.array(labels)[y]
    model = pdc().fit(X, mapped)
    assert model.decide_use_prior() == (path == 'prior')
    pred = model.predict(X)
    assert (pred == mapped).mean() > 0.8


def test_pandas_categorical_target(iris):
    X, y = iris
    target = pd.Series(pd.Categorical.from_codes(y, ['setosa', 'versicolor', 'virginica']))
    assert set(pdc().fit(X, target).predict(X[:5])) == {'setosa'}


def test_column_vector_target_warns_and_works(iris):
    X, y = iris
    with pytest.warns(DataConversionWarning):
        model = pdc().fit(X, y.reshape(-1, 1))
    assert model.score(X, y) > 0.9


def test_single_class_is_rejected(iris):
    with pytest.raises(ValueError, match='one class'):
        pdc().fit(iris[0][:10], np.zeros(10))


def test_continuous_target_is_rejected(iris):
    with pytest.raises(ValueError, match='Unknown label type'):
        pdc().fit(iris[0], iris[0][:, 0])


# --------------------------------------------------------------------------- feature inputs

def test_numpy_and_dataframe_inputs_are_interchangeable(cancer_frame):
    X_train, X_test, y_train, _ = cancer_frame
    reference = pdc().fit(X_train.to_numpy(), y_train.to_numpy()).predict_proba(X_test.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # scikit-learn's feature-name warnings
        np.testing.assert_allclose(pdc().fit(X_train.to_numpy(), y_train).predict_proba(X_test), reference)
        np.testing.assert_allclose(pdc().fit(X_train, y_train).predict_proba(X_test.to_numpy()), reference)
    np.testing.assert_allclose(pdc().fit(X_train, y_train).predict_proba(X_test), reference)


def test_index_of_X_and_y_does_not_matter(cancer_frame):
    X_train, X_test, y_train, _ = cancer_frame
    reference = pdc().fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True)).predict_proba(X_test)
    X_odd = X_train.set_axis([f'row{i}' for i in range(len(X_train))])
    y_other = y_train.set_axis(np.arange(len(y_train))[::-1])  # y index unrelated to X index
    np.testing.assert_allclose(pdc().fit(X_odd, y_other).predict_proba(X_test), reference)
    shuffled = X_test.sample(frac=1, random_state=0)
    np.testing.assert_allclose(pdc().fit(X_train, y_train).predict_proba(shuffled), reference[X_test.index.get_indexer(shuffled.index)])


def test_bool_uint_nullable_and_float32_columns(cancer_frame):
    X_train, X_test, y_train, _ = cancer_frame

    def mixed(X):
        X = X.copy()
        X['flag'] = X.iloc[:, 0] > 14
        X['small'] = (X.iloc[:, 1] // 5).astype(np.uint8)
        X['nullable'] = pd.array(np.where(X.iloc[:, 2] > 90, 1, 0), dtype='Int64')
        X['f32'] = X.iloc[:, 3].astype(np.float32)
        return X

    model = pdc().fit(mixed(X_train), y_train)
    assert model.n_features_in_ == X_train.shape[1] + 4
    assert model.X_anchors_.dtype == np.float64
    assert model.score(mixed(X_test), cancer_frame[3]) > 0.85
    # uint differences must not wrap around
    assert np.abs(model.X_anchors_).max() < 1e4


def test_float32_input_stays_float32(iris):
    X, y = iris
    assert pdc().fit(X.astype(np.float32), y).X_anchors_.dtype == np.float32


def titanic_like(n=200, seed=0):
    rng = np.random.RandomState(seed)
    sex = rng.choice(['male', 'female'], n)
    klass = pd.Categorical(rng.choice(['1st', '2nd', '3rd'], n), categories=['1st', '2nd', '3rd'], ordered=True)
    age = rng.uniform(1, 70, n)
    age[rng.rand(n) < 0.1] = np.nan
    embarked = rng.choice(['S', 'C', 'Q', None], n, p=[.6, .2, .15, .05])
    survived = ((sex == 'female') & (klass.codes < 2)) | (rng.rand(n) < 0.15)
    X = pd.DataFrame({'sex': sex, 'class': klass, 'age': age, 'embarked': embarked, 'fare': rng.gamma(2, 20, n)})
    return X, survived.astype(int)


def test_string_and_categorical_columns_are_encoded():
    X, y = titanic_like()
    model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=30, random_state=0)).fit(X[:150], y[:150])
    assert model.score(X[150:], y[150:]) > 0.7
    # the ordered categorical keeps its order, unseen and missing values become -1
    assert model._categories['class'] == ['1st', '2nd', '3rd']
    X_new = X[150:155].copy()
    X_new['embarked'] = ['Z', None, 'S', 'C', 'Q']
    assert model.predict(X_new).shape == (5,)
    with pytest.warns(UserWarning, match='feature names'):
        assert model.predict(X[150:155].to_numpy(dtype=object)).shape == (5,)  # same layout without column names


def test_identifier_like_columns_trigger_a_warning():
    X, y = titanic_like()
    X['name'] = [f'passenger {i}' for i in range(len(X))]
    with pytest.warns(UserWarning, match="look like identifiers"):
        PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=5)).fit(X, y)


def test_datetime_column_is_rejected_with_advice():
    X, y = titanic_like()
    X['date'] = pd.date_range('2020-01-01', periods=len(X))
    with pytest.raises(TypeError, match='Convert it first'):
        pdc().fit(X, y)


def test_string_numpy_array_gets_actionable_error():
    X = np.array([['a', 1.], ['b', 2.], ['a', 3.], ['b', 4.]], dtype=object)
    with pytest.raises(ValueError, match='pandas DataFrame'):
        pdc().fit(X, [0, 1, 0, 1])


def test_nan_is_accepted_only_if_the_base_estimator_supports_it(cancer_frame):
    X_train, X_test, y_train, _ = cancer_frame
    X_nan = X_train.copy()
    X_nan.iloc[::7, 0] = np.nan
    PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=20)).fit(X_nan, y_train).predict(X_test)
    with pytest.raises(ValueError, match='NaN'), warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        PairwiseDifferenceClassifier(LogisticRegression()).fit(X_nan, y_train)


# --------------------------------------------------------------------------- estimator handling

def test_base_estimator_is_cloned_not_modified(iris):
    X, y = iris
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    before = rf.predict(X)
    PairwiseDifferenceClassifier(rf).fit(X, y)
    np.testing.assert_array_equal(rf.predict(X), before)  # 0.3.1 refit rf on 12 pair features


def test_refit_recomputes_the_prior(iris):
    X, y = iris
    model = pdc().fit(X, y)
    np.testing.assert_allclose(model.prior_, [1 / 3] * 3)
    model.fit(X[:60], y[:60])
    np.testing.assert_allclose(model.prior_, [50 / 60, 10 / 60])


def test_errors_and_not_fitted(iris):
    X, y = iris
    with pytest.raises(NotFittedError):
        PairwiseDifferenceClassifier().predict(X)
    with pytest.raises(TypeError, match='instance'):
        PairwiseDifferenceClassifier(RandomForestClassifier).fit(X, y)
    with pytest.raises(ValueError, match='use_prior'):
        pdc(use_prior='sometimes').fit(X, y)
    assert pdc(use_prior=np.True_).fit(X, y).decide_use_prior() is True  # numpy bools, e.g. from a GridSearch grid
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match='max_pairs'):
            pdc(max_pairs=bad).fit(X, y)


def test_parametric_base_estimator_warns(iris):
    with pytest.warns(UserWarning, match='parametric'):
        PairwiseDifferenceClassifier(LogisticRegression(max_iter=500)).fit(*iris)


def test_default_estimator_and_reproducibility(iris):
    X, y = iris
    first = PairwiseDifferenceClassifier(random_state=3).fit(X, y)
    assert type(first.estimator_).__name__ == 'ExtraTreesClassifier'
    assert first.estimator is None
    np.testing.assert_array_equal(first.predict_proba(X), PairwiseDifferenceClassifier(random_state=3).fit(X, y).predict_proba(X))


def test_works_with_sklearn_tooling(iris):
    X, y = iris
    assert cross_val_score(pdc(), X, y, cv=3).mean() > 0.85
    search = GridSearchCV(pdc(), {'estimator__max_depth': [2, None], 'use_prior': [True, False]}, cv=3).fit(X, y)
    assert search.best_score_ > 0.85
    assert make_pipeline(StandardScaler(), pdc()).fit(X, y).score(X, y) > 0.9
    model = pdc().fit(X, y)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(X), model.predict(X))
    assert clone(model).get_params()['estimator'].get_params() == forest().get_params()


# --------------------------------------------------------------------------- the maths, with a deterministic base estimator

def test_training_pairs_follow_the_paper(iris):
    X, y = iris[0][::25], iris[1][::25]
    model = PairwiseDifferenceClassifier(FakePairClassifier()).fit(X, y)
    expected_X = np.array([np.concatenate([X[i], X[j], X[i] - X[j]]) for i in range(len(X)) for j in range(len(X))])
    expected_y = np.array([int(y[i] != y[j]) for i in range(len(X)) for j in range(len(X))])
    np.testing.assert_array_equal(model.estimator_.X_fit_, expected_X)
    np.testing.assert_array_equal(model.estimator_.y_fit_, expected_y)


def _literal_proba(model, X, use_prior):
    """Straight transcription of the paper (prior path) and of the mean-similarity rule, one loop per term."""
    fake = model.estimator_
    anchors, anchor_class, prior = model.X_anchors_, model.y_anchors_, model.prior_
    out = []
    for x in X:
        sims = []
        for a in anchors:
            p_ab = fake.predict_proba(np.concatenate([x, a, x - a])[None])[0, 0]
            p_ba = fake.predict_proba(np.concatenate([a, x, a - x])[None])[0, 0]
            sims.append((p_ab + p_ba) / 2)
        sims = np.array(sims)
        if use_prior:
            post = np.zeros(len(prior))
            for s, c_anchor in zip(sims, anchor_class):
                for c in range(len(prior)):
                    post[c] += (s if c == c_anchor else prior[c] * (1 - s) / (1 - prior[c_anchor])) / len(anchors)
        else:
            post = np.array([sims[anchor_class == c].mean() for c in range(len(prior))])
        out.append(post / post.sum())
    return np.array(out)


@pytest.mark.parametrize('use_prior', [True, False])
def test_aggregation_matches_the_literal_formula(iris, use_prior):
    X, y = iris[0][::6], iris[1][::6]
    X_test = iris[0][3::17]
    model = PairwiseDifferenceClassifier(FakePairClassifier(), use_prior=use_prior).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X_test), _literal_proba(model, X_test, use_prior), rtol=1e-12)


def test_per_anchor_posteriors_average_to_the_prior_path(iris):
    X, y = iris[0][::6], iris[1][::6]
    model = PairwiseDifferenceClassifier(FakePairClassifier(), use_prior=True).fit(X, y)
    samples = model.predict_proba_samples(X[:7])
    assert samples.shape == (7, len(X), 3)
    np.testing.assert_allclose(samples.sum(axis=2), 1.)
    mean = samples.mean(axis=1)
    np.testing.assert_allclose(mean / mean.sum(axis=1, keepdims=True), model.predict_proba(X[:7]))
    assert 0. <= model.score_difference(X[:7], y[:7]) <= 1.


def test_chunked_prediction_gives_the_same_result(iris, monkeypatch):
    X, y = iris
    model = pdc().fit(X, y)
    reference = model.predict_proba(X)
    monkeypatch.setattr(pdll._base, 'CHUNK_CELLS', 50)  # one test row per chunk
    np.testing.assert_array_equal(model.predict_proba(X), reference)


# --------------------------------------------------------------------------- subsampling

def test_max_pairs_subsamples_training_pairs(iris):
    X, y = iris
    model = PairwiseDifferenceClassifier(FakePairClassifier(), max_pairs=1000, random_state=0).fit(X, y)
    pairs = model.estimator_.X_fit_
    assert len(pairs) == 1000
    assert len(np.unique(pairs, axis=0)) == 1000  # sampled without replacement
    assert set(model.estimator_.y_fit_) == {0, 1}
    again = PairwiseDifferenceClassifier(FakePairClassifier(), max_pairs=1000, random_state=0).fit(X, y)
    np.testing.assert_array_equal(again.estimator_.X_fit_, pairs)


def test_max_anchors_keeps_every_class(iris):
    X, y = iris[0], np.where(np.arange(150) < 140, 0, 1)  # 140 vs 10 rows
    model = pdc(max_anchors=30, random_state=0).fit(X, y)
    assert len(model.X_anchors_) == 30
    assert set(model.y_anchors_) == {0, 1}
    np.testing.assert_allclose(model.prior_, [140 / 150, 10 / 150])  # prior from the full training set


def test_large_dataset_stays_fast():
    from time import perf_counter
    rng = np.random.RandomState(0)
    X = rng.normal(size=(5000, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    start = perf_counter()
    model = PairwiseDifferenceClassifier(ExtraTreesClassifier(n_estimators=10, class_weight='balanced', n_jobs=-1, random_state=0),
                                         max_pairs=100_000, max_anchors=200, random_state=0).fit(X, y)
    assert model.score(X[:1000], y[:1000]) > 0.85
    assert perf_counter() - start < 60


# --------------------------------------------------------------------------- sample weights

def test_integer_weights_equal_repeated_rows(iris):
    X, y = iris[0][::10], iris[1][::10]
    weights = np.arange(len(y)) % 3 + 1
    weighted = PairwiseDifferenceClassifier(FakePairClassifier(), use_prior=True).fit(X, y, sample_weight=weights)
    repeated = PairwiseDifferenceClassifier(FakePairClassifier(), use_prior=True).fit(np.repeat(X, weights, axis=0), np.repeat(y, weights))
    np.testing.assert_allclose(weighted.predict_proba(iris[0]), repeated.predict_proba(iris[0]), rtol=1e-12)
    np.testing.assert_allclose(weighted.prior_, repeated.prior_)
    pair_weights = weighted.estimator_.sample_weight_fit_
    np.testing.assert_array_equal(pair_weights, np.outer(weights, weights).ravel())


@pytest.mark.parametrize('use_prior', [True, False])
def test_zero_weight_equals_removing_the_row(iris, use_prior):
    X, y = iris[0][::5], iris[1][::5]
    weights = np.ones(len(y))
    weights[::4] = 0
    weighted = PairwiseDifferenceClassifier(FakePairClassifier(), use_prior=use_prior).fit(X, y, sample_weight=weights)
    removed = PairwiseDifferenceClassifier(FakePairClassifier(), use_prior=use_prior).fit(X[weights > 0], y[weights > 0])
    np.testing.assert_allclose(weighted.predict_proba(iris[0]), removed.predict_proba(iris[0]), rtol=1e-12)


def test_sample_weight_needs_base_estimator_support(iris):
    with pytest.raises(ValueError, match='does not support sample_weight'):
        PairwiseDifferenceClassifier(KNeighborsClassifier()).fit(*iris, sample_weight=np.ones(150))
    with pytest.raises(ValueError, match='Negative'):
        pdc().fit(*iris, sample_weight=-np.ones(150))


# --------------------------------------------------------------------------- found by the real-world campaign (benchmark/real_world)

def test_unseen_categories_raise_no_pandas_deprecation():
    """pd.Categorical with values outside its categories is deprecated in pandas 3 and will raise in pandas 4."""
    X, y = titanic_like()
    model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=5)).fit(X, y)
    X_new = X[:4].copy()
    X_new['embarked'] = ['Z', 'Y', None, 'S']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.predict(X_new)
    deprecations = [w for w in caught if issubclass(w.category, (DeprecationWarning, FutureWarning)) or 'Pandas' in w.category.__name__]
    # old scikit-learn releases raise their own deprecations against newer pandas: only pdll's code and Categorical matter here
    from_pdll = [w for w in deprecations if 'Categorical' in str(w.message) or os.path.basename(os.path.dirname(w.filename)) == 'pdll']
    assert not from_pdll, [str(w.message) for w in from_pdll]


def test_actionable_errors_for_multi_output_and_missing_target(iris):
    X, y = iris
    with pytest.raises(ValueError, match='MultiOutputClassifier'):
        pdc().fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match='contains NaN.*Drop the rows'):
        pdc().fit(X, np.where(np.arange(150) % 7 == 0, np.nan, y))


def test_auto_max_pairs_rule():
    resolve = PairwiseDifferenceClassifier._resolve_max_pairs
    assert resolve(None, 2000, 10) is None and resolve(5000, 2000, 10) == 5000  # explicit values are kept
    assert resolve('auto', 300, 10) == pdll._base.DEFAULT_MAX_PAIRS  # 300 rows: every pair is below the cap
    wide = pdll._base.PAIR_CELL_BUDGET // (3 * 2000)
    assert resolve('auto', 300, 2000) == wide < pdll._base.DEFAULT_MAX_PAIRS  # wide data: the memory budget binds
    with pytest.warns(UserWarning, match='memory budget'):
        resolve('auto', 5000, 20_000)  # fewer pairs than rows: tell the user


def test_auto_max_pairs_in_fit(iris):
    X = np.tile(iris[0], (3, 1)) + np.random.RandomState(0).normal(scale=0.01, size=(450, 4))
    y = np.tile(iris[1], 3)
    small = PairwiseDifferenceClassifier(FakePairClassifier()).fit(X[:300], y[:300])
    assert len(small.estimator_.X_fit_) == 300 * 300  # all pairs, as in the paper
    large = PairwiseDifferenceClassifier(FakePairClassifier(), random_state=0).fit(X, y)
    assert len(large.estimator_.X_fit_) == pdll._base.DEFAULT_MAX_PAIRS
    assert len(large.X_anchors_) == 450  # max_anchors=500
    with pytest.raises(ValueError, match='auto'):
        pdc(max_pairs='many').fit(*iris)


def test_class_weight_is_applied_as_sample_weight(iris):
    from sklearn.utils.class_weight import compute_sample_weight
    X, y = iris[0][:120], iris[1][:120]  # 50 / 50 / 20: imbalanced
    balanced = PairwiseDifferenceClassifier(FakePairClassifier(), class_weight='balanced').fit(X, y)
    weighted = PairwiseDifferenceClassifier(FakePairClassifier()).fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    np.testing.assert_allclose(balanced.predict_proba(iris[0]), weighted.predict_proba(iris[0]))
    np.testing.assert_allclose(balanced.prior_, [1 / 3] * 3)
    dropped = PairwiseDifferenceClassifier(FakePairClassifier(), class_weight={0: 1, 1: 1, 2: 0}).fit(X, y)
    np.testing.assert_array_equal(dropped.classes_, [0, 1])  # like rows with a zero sample_weight, the class is left out
    np.testing.assert_allclose(dropped.prior_, [0.5, 0.5])
    with pytest.raises(ValueError, match='class_weight is applied through sample weights'):
        PairwiseDifferenceClassifier(KNeighborsClassifier(), class_weight='balanced').fit(X, y)


class RecordingForest(ExtraTreesClassifier):
    """ExtraTreesClassifier that records the labels, sample weights and class_weight its fit receives."""

    def fit(self, X, y, sample_weight=None):
        self.fit_args_ = (np.asarray(y), None if sample_weight is None else np.asarray(sample_weight), self.class_weight)
        return super().fit(X, y, sample_weight=sample_weight)


def test_balanced_base_estimator_balances_weighted_pairs(iris):
    # Forests before scikit-learn 1.8 compute class_weight="balanced" from row counts and ignore sample_weight, so PDL
    # balances "same" against "different" pairs on the weighted counts itself: every version then fits the same model.
    X, y = iris[0][:120], iris[1][:120]
    w = np.random.RandomState(0).uniform(0.1, 10, len(y))
    weighted = PairwiseDifferenceClassifier(RecordingForest(n_estimators=5, class_weight='balanced', random_state=0))
    weighted.fit(X, y, sample_weight=w)
    y_pairs, pair_weight, class_weight = weighted.estimator_.fit_args_
    assert class_weight is None
    np.testing.assert_allclose(pair_weight[y_pairs == 0].sum(), pair_weight[y_pairs == 1].sum())
    assert weighted.estimator_.class_weight == 'balanced'  # the fitted clone still shows the user's setting
    unweighted = PairwiseDifferenceClassifier(RecordingForest(n_estimators=5, class_weight='balanced', random_state=0))
    unweighted.fit(X, y)
    assert unweighted.estimator_.fit_args_[1:] == (None, 'balanced')  # no weights: scikit-learn balances, as in the paper


def test_large_training_sets_get_a_warning():
    rng = np.random.RandomState(0)
    X, y = rng.normal(size=(2500, 3)), rng.randint(0, 2, 2500)
    with pytest.warns(UserWarning, match='designed for small datasets'):
        PairwiseDifferenceClassifier(FakePairClassifier(), max_pairs=1000, max_anchors=50).fit(X, y)
