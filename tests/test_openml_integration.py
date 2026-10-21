"""Real public datasets, downloaded from OpenML and used raw: mixed dtypes, missing values, string targets.

Needs network access; run with ``pytest -m network``.
"""
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from helpers import check_score
from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

pytestmark = pytest.mark.network


def fetch(name, version):
    from sklearn.datasets import fetch_openml
    try:
        return fetch_openml(name=name, version=version, as_frame=True, return_X_y=True)
    except Exception as error:  # noqa: BLE001 - OpenML outages must not look like pdll bugs
        pytest.skip(f'OpenML unavailable: {error}')


@pytest.mark.parametrize('name, version, reference', [
    ('credit-g', 1, 0.7167),   # 1000 rows, 13 categorical + 7 numeric columns, string target
    ('titanic', 1, 0.79),    # missing values, names and tickets (identifier-like strings)
])
def test_classifier_on_raw_openml_dataframe(name, version, reference):
    X, y = fetch(name, version)
    X = X.drop(columns=[c for c in ('boat', 'body', 'home.dest', 'name', 'ticket', 'cabin') if c in X.columns])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=400, test_size=300, random_state=0, stratify=y)
    model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=50, class_weight='balanced', random_state=0))
    model.fit(X_train, y_train)
    assert set(model.predict(X_test)) <= set(y.unique())
    check_score(model.score(X_test, y_test), reference)


def test_regressor_on_raw_openml_dataframe():
    X, y = fetch('cholesterol', 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = PairwiseDifferenceRegressor(HistGradientBoostingRegressor(max_iter=50, random_state=0)).fit(X_train, y_train)
    assert np.isfinite(model.predict(X_test)).all()  # R2 is about 0.05 here: too noisy for a score floor
