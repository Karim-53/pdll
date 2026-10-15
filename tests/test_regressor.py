"""Unit tests of PairwiseDifferenceRegressor and PDCDataTransformer."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from helpers import FakePairRegressor
from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor, PDCDataTransformer
from test_classifier import titanic_like


@pytest.fixture(scope='module')
def diabetes():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X[:120], y[:120], X[120:180], y[120:180]


def pdr(**kwargs):
    return PairwiseDifferenceRegressor(ExtraTreesRegressor(n_estimators=20, random_state=0), **kwargs)


def test_predict_is_ndarray_and_decent(diabetes):
    X, y, X_test, y_test = diabetes
    pred = pdr().fit(X, y).predict(X_test)
    assert isinstance(pred, np.ndarray) and pred.shape == (len(X_test),)
    assert pdr().fit(X, y).score(X_test, y_test) > 0.2


def test_training_pairs_and_symmetric_prediction(diabetes):
    X, y = diabetes[0].to_numpy()[:6], diabetes[1].to_numpy()[:6]
    model = PairwiseDifferenceRegressor(FakePairRegressor()).fit(X, y)
    expected = np.array([np.concatenate([X[i], X[j], X[i] - X[j]]) for i in range(6) for j in range(6)])
    np.testing.assert_array_equal(model.estimator_.X_fit_, expected)
    np.testing.assert_allclose(model.estimator_.y_fit_, [y[i] - y[j] for i in range(6) for j in range(6)])
    x = diabetes[2].to_numpy()[:1]
    literal = np.mean([y[j] + (FakePairRegressor().predict(np.concatenate([x[0], X[j], x[0] - X[j]])[None])[0]
                               - FakePairRegressor().predict(np.concatenate([X[j], x[0], X[j] - x[0]])[None])[0]) / 2 for j in range(6)])
    np.testing.assert_allclose(model.predict(x), [literal])


def test_inputs_index_dtypes_and_categoricals(diabetes):
    X, y, X_test, _ = diabetes
    reference = pdr().fit(X.reset_index(drop=True), y.reset_index(drop=True)).predict(X_test)
    np.testing.assert_allclose(pdr().fit(X.set_axis(X.index[::-1]), y).predict(X_test), reference)
    X_mixed = X.assign(sex=np.where(X['sex'] > 0, 'm', 'f'))
    model = pdr().fit(X_mixed, y.astype(np.uint16))  # uint target: differences must not wrap around
    assert model.y_anchors_.dtype == np.float64
    assert model.predict(X_test.assign(sex=np.where(X_test['sex'] > 0, 'm', 'f'))).min() > 0


def test_base_estimator_not_modified_and_errors(diabetes):
    X, y, X_test, _ = diabetes
    tree = DecisionTreeRegressor(random_state=0).fit(X, y)
    before = tree.predict(X_test)
    PairwiseDifferenceRegressor(tree).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X_test), before)
    with pytest.raises(NotFittedError):
        PairwiseDifferenceRegressor().predict(X_test)
    with pytest.raises(ValueError, match='max_anchors'):
        pdr(max_anchors=0).fit(X, y)


def test_nan_with_hist_gradient_boosting(diabetes):
    X, y, X_test, _ = diabetes
    X = X.copy()
    X.iloc[::5, 2] = np.nan
    assert np.isfinite(PairwiseDifferenceRegressor(HistGradientBoostingRegressor(max_iter=20)).fit(X, y).predict(X_test)).all()


def test_sample_weight_semantics(diabetes):
    X, y, X_test, _ = diabetes
    X, y = X.to_numpy()[:20], y.to_numpy()[:20]
    weights = np.arange(20) % 3
    weighted = PairwiseDifferenceRegressor(FakePairRegressor()).fit(X, y, sample_weight=weights)
    keep = weights > 0
    repeated = PairwiseDifferenceRegressor(FakePairRegressor()).fit(np.repeat(X[keep], weights[keep], axis=0), np.repeat(y[keep], weights[keep]))
    X_test = X_test.to_numpy()
    np.testing.assert_allclose(weighted.predict(X_test), repeated.predict(X_test), rtol=1e-12)
    np.testing.assert_allclose(weighted.sample_weight_.sum(), 1.)


def test_subsampling(diabetes):
    X, y, X_test, _ = diabetes
    model = PairwiseDifferenceRegressor(FakePairRegressor(), max_pairs=500, max_anchors=25, random_state=0).fit(X, y)
    assert len(model.estimator_.X_fit_) == 500 and len(model.X_anchors_) == 25
    assert model.predict(X_test).shape == (len(X_test),)


@pytest.mark.parametrize('method', ['L2', 'KLD', 'Optimize', 'L1L2', 'L1', 'ExtremeWeightPruning', 'NegativeError',
                                    'InverseError', 'OrderedVoting', 'KMeansClusterCenters'])
def test_anchor_weighting_methods(diabetes, method):
    X, y, X_val, y_val = diabetes
    model = pdr().fit(X[:50], y[:50])
    model.learn_anchor_weights(X_val, y_val, method=method, enable_warnings=False)
    weights = model.sample_weight_
    assert isinstance(weights, pd.Series) and weights.index.equals(X.index[:50])
    assert weights.notna().all() and (weights >= -1e-12).all()
    before = weights.copy()
    model.predict(X_val)
    pd.testing.assert_series_equal(model.sample_weight_, before)  # 0.3.1's predict() wrote NaN into the weights


def test_set_sample_weight_and_unknown_method(diabetes):
    X, y, X_val, y_val = diabetes
    model = pdr().fit(X[:30], y[:30])
    model.set_sample_weight(np.linspace(0, 1, 30))
    assert model.sample_weight_.index.equals(X.index[:30])
    with pytest.raises(ValueError, match='same index'):
        model.set_sample_weight(pd.Series(np.ones(30), index=range(100, 130)))
    with pytest.raises(ValueError, match='unknown'):
        model.learn_anchor_weights(X_val, y_val, method='Magic')


def test_works_with_cross_validation(diabetes):
    X, y, _, _ = diabetes
    assert cross_val_score(pdr(), X, y, cv=3).mean() > 0.1


# --------------------------------------------------------------------------- PDCDataTransformer

def test_data_transformer_is_self_contained_and_pipelines():
    X, y = titanic_like()
    X = X.drop(columns='age')
    transformer = PDCDataTransformer().fit(X)
    assert transformer.numeric_features_ == ['fare'] and transformer.ordinal_features_ == ['class']
    Xt = transformer.transform(X)
    assert Xt.dtype == np.float32 and Xt.shape[0] == len(X)
    assert transformer.numeric_features is None  # fit does not modify parameters
    pipeline = make_pipeline(PDCDataTransformer(), PairwiseDifferenceClassifier(DecisionTreeClassifier(class_weight='balanced', random_state=0)))
    assert pipeline.fit(X[:150], y[:150]).score(X[150:], y[150:]) > 0.6
    _, y_t = PDCDataTransformer(y_type='ordinal').fit(X, y).transform(X, y)
    assert y_t.shape == (len(X), 1)
    with pytest.raises(ValueError, match='y_type'):
        PDCDataTransformer(y_type='bad').fit(X)
