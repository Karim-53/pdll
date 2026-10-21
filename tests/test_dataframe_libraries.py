"""polars and pyarrow tables must give exactly what the same pandas DataFrame gives (skipped when not installed)."""
import importlib

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
from test_classifier import titanic_like


def import_or_skip(name):
    try:
        return importlib.import_module(name)
    except ImportError as error:  # not installed, or built for another numpy (recent pyarrow needs numpy 2)
        pytest.skip(f'{name} unavailable: {error}')


def to_library(library, X, y):
    lib = import_or_skip(library)
    import_or_skip('pyarrow')  # polars converts to pandas through pyarrow
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


NO_PYARROW_SCRIPT = '''
import sys


class BlockPyarrow:
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("No module named 'pyarrow'")


sys.meta_path.insert(0, BlockPyarrow())
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from pdll import PairwiseDifferenceClassifier

data = {"sex": ["m", "f", "f"] * 30, "age": np.arange(90.0), "n": list(range(90))}
labels = [0, 1, 1] * 30
model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=10, random_state=0))
expected = model.fit(pd.DataFrame(data), labels).predict_proba(pd.DataFrame(data))
got = model.fit(pl.DataFrame(data), pl.Series(labels)).predict_proba(pl.DataFrame(data))
assert "pyarrow" not in sys.modules
np.testing.assert_allclose(got, expected)
print("ok", list(model.feature_names_in_))
'''


def test_polars_works_without_pyarrow():
    """polars' to_pandas needs pyarrow, which neither pdll nor polars installs: pdll falls back to polars' to_numpy."""
    import os
    import subprocess
    import sys
    import_or_skip('polars')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    done = subprocess.run([sys.executable, '-c', NO_PYARROW_SCRIPT], env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok ['sex', 'age', 'n']"


def test_polars_lazyframe_is_collected(titanic):
    pl = import_or_skip('polars')
    import_or_skip('pyarrow')
    X, y = titanic
    model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=10, random_state=0))
    expected = model.fit(X, y).predict_proba(X[:20])
    np.testing.assert_allclose(model.fit(pl.from_pandas(X).lazy(), y).predict_proba(pl.from_pandas(X[:20]).lazy()), expected)


def test_pyarrow_dates_are_rejected_like_pandas_datetimes():
    """pyarrow date32 columns must not slip through as datetime.date objects encoded as categories."""
    pa = import_or_skip('pyarrow')
    import datetime
    dates = [datetime.date(2020, 1 + i % 12, 1) for i in range(60)]
    table = pa.table({'when': pa.array(dates, pa.date32()), 'x': np.arange(60.)})
    with pytest.raises(TypeError, match='Convert it first'):
        PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=5)).fit(table, [0, 1] * 30)


def test_transformer_on_polars_matches_pandas(titanic):
    pl = import_or_skip('polars')
    import_or_skip('pyarrow')
    from pdll import PDCDataTransformer
    X, _ = titanic
    X = X.drop(columns='age')  # the transformer does not impute
    expected = PDCDataTransformer().fit_transform(X)
    np.testing.assert_allclose(PDCDataTransformer().fit_transform(pl.from_pandas(X)), expected)


def test_polars_boolean_target_with_nulls_gets_the_missing_target_message():
    pl = import_or_skip('polars')
    import_or_skip('pyarrow')
    X = np.arange(40.).reshape(20, 2)
    with pytest.raises(ValueError, match='Drop the rows whose target is missing'):
        PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=5)).fit(X, pl.Series([True, False] * 9 + [None, True]))


def test_other_objects_with_to_pandas_are_left_to_scikit_learn():
    """Only polars and pyarrow data is converted; any other array-like goes to scikit-learn's validation as before."""
    class ArrayLike:
        def __init__(self, values):
            self.values = values

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.values, dtype=dtype)

        def to_pandas(self):  # means something else here
            import pandas as pd
            return pd.Index(['summary'])

    X, y = np.random.RandomState(0).normal(size=(60, 2)), np.arange(60) % 2
    model = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=5, random_state=0))
    np.testing.assert_allclose(model.fit(ArrayLike(X), y).predict_proba(X), model.fit(X, y).predict_proba(X))
