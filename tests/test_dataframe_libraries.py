"""polars and pyarrow tables must give exactly what the same pandas DataFrame gives (skipped when not installed)."""
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
from test_classifier import titanic_like


def to_library(library, X, y):
    lib = pytest.importorskip(library)
    pytest.importorskip('pyarrow')  # polars converts to pandas through pyarrow
    if library == 'polars':
        return lib.from_pandas(X), lib.Series(y)
    return lib.Table.from_pandas(X, preserve_index=False), lib.array(y)


@pytest.fixture
def titanic():
    X, y = titanic_like()
    return X.assign(**{'class': X['class'].astype(str)}), y  # plain strings: polars and pyarrow have no ordered categoricals


@pytest.mark.parametrize('library', ['polars', 'pyarrow'])
def test_classifier_on_polars_and_pyarrow_tables(library, titanic):
    X, y = titanic
    table, labels = to_library(library, X, y)
    expected = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=30, random_state=0)).fit(X[:150], y[:150])
    model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=30, random_state=0)).fit(table[:150], labels[:150])
    np.testing.assert_allclose(model.predict_proba(table[150:]), expected.predict_proba(X[150:]))
    assert list(model.feature_names_in_) == list(X.columns)


@pytest.mark.parametrize('library', ['polars', 'pyarrow'])
def test_regressor_on_polars_and_pyarrow_tables(library, titanic):
    X, _ = titanic
    X, y = X.drop(columns='fare'), X['fare'].to_numpy()
    table, target = to_library(library, X, y)
    expected = PairwiseDifferenceRegressor(HistGradientBoostingRegressor(max_iter=30, random_state=0)).fit(X[:150], y[:150])
    model = PairwiseDifferenceRegressor(HistGradientBoostingRegressor(max_iter=30, random_state=0)).fit(table[:150], target[:150])
    np.testing.assert_allclose(model.predict(table[150:]), expected.predict(X[150:]))
