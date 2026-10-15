"""Turn user input (numpy, lists, pandas with mixed dtypes) into the float arrays PDL works on."""
import numbers
import warnings

import numpy as np
import pandas as pd

from ._compat import validate_data

# A string column with more distinct values than this share of its rows is probably an identifier (name, ticket, ...).
ID_LIKE_UNIQUE_RATIO = 0.5
ID_LIKE_MIN_ROWS = 20

_NUMERIC_INFERRED_TYPES = ('integer', 'floating', 'mixed-integer-float', 'decimal', 'boolean', 'empty')
_STRING_HINT = ('PDL needs numeric features. Pass a pandas DataFrame (pdll encodes string and categorical columns '
                'itself) or encode them first, e.g. with sklearn.preprocessing.OrdinalEncoder.')


def _is_plain_numeric(dtype) -> bool:
    """numpy int, uint and float dtypes (no bool, no pandas extension dtypes)."""
    return isinstance(dtype, np.dtype) and dtype.kind in 'iuf'


def _check_supported_dtype(column, dtype):
    if (pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype)
            or isinstance(dtype, (pd.PeriodDtype, pd.IntervalDtype))):
        raise TypeError(f'Column {column!r} has dtype {dtype}, but PDL needs numbers. Convert it first, e.g. '
                        f'extract the year/month/day or use .astype("int64").')


def learn_categories(X: pd.DataFrame) -> dict:
    """Find the columns of X that must be encoded and fix their categories (position = code)."""
    categories, id_like = {}, []
    for column in X.columns:
        series = X[column]
        dtype = series.dtype
        _check_supported_dtype(column, dtype)
        if isinstance(dtype, pd.CategoricalDtype):
            categories[column] = list(dtype.categories)  # keeps the order of ordered categoricals
        elif pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            continue
        elif pd.api.types.infer_dtype(series, skipna=True) in _NUMERIC_INFERRED_TYPES:
            continue  # object column holding numbers
        else:
            values = series.dropna()
            categories[column] = sorted(pd.unique(values.to_numpy(dtype=object)), key=str)
            if len(values) >= ID_LIKE_MIN_ROWS and len(categories[column]) > ID_LIKE_UNIQUE_RATIO * len(values):
                id_like.append(column)
    if id_like:
        warnings.warn(f'Columns {id_like} look like identifiers (mostly distinct values). PDL encodes them as categories, '
                      f'which rarely helps and slows training; consider dropping them.', UserWarning, stacklevel=4)
    return categories


def encode_frame(X: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Numeric copy of X with the same column names.

    Categorical columns become their category code (-1 = missing or unseen), bool and nullable columns become float64.
    float32 is kept only when every column is float32.
    """
    arrays = []
    for column in X.columns:
        series = X[column]
        if column in categories:
            # -1 for missing and unseen values (pd.Categorical with unseen values is deprecated in pandas 3)
            codes = pd.Index(categories[column], dtype=object).get_indexer(series.to_numpy(dtype=object))
            arrays.append(codes.astype(np.float64))
        elif _is_plain_numeric(series.dtype):
            arrays.append(series.to_numpy())
        else:
            _check_supported_dtype(column, series.dtype)
            try:
                arrays.append(series.to_numpy(dtype=np.float64, na_value=np.nan))
            except (TypeError, ValueError) as error:
                raise ValueError(f'Column {column!r} could not be converted to numbers ({error}). {_STRING_HINT}') from error
    dtype = np.float32 if arrays and all(a.dtype == np.float32 for a in arrays) else np.float64
    values = np.column_stack([a.astype(dtype, copy=False) for a in arrays]) if arrays else np.empty((len(X), 0), dtype)
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def needs_encoding(X: pd.DataFrame) -> bool:
    return not all(_is_plain_numeric(dtype) for dtype in X.dtypes)


def _shape(y):
    """Shape of y without calling numpy functions on it (array-likes may forbid them); () when absent."""
    if y is None or (isinstance(y, str) and y == 'no_validation'):
        return ()
    shape = getattr(y, 'shape', None)
    return tuple(shape) if shape is not None else np.asarray(y).shape


def to_float_array(estimator, X, y='no_validation', *, reset, allow_nan, **check_params):
    """Validate X (and y) like scikit-learn does and return X as a float32/float64 ndarray.

    On ``reset`` (fit) the categorical columns of a DataFrame are learned and stored in ``estimator._categories``;
    afterwards the same encoding is applied, by column name, or by position if X is no longer a DataFrame.
    """
    if reset:
        estimator._categories = learn_categories(X) if isinstance(X, pd.DataFrame) else {}
        estimator._columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
    categories = estimator._categories
    if isinstance(X, pd.DataFrame):
        if categories or needs_encoding(X):
            X = encode_frame(X, categories)
    elif categories and estimator._columns is not None and np.ndim(X) == 2 and np.shape(X)[1] == len(estimator._columns):
        # fitted on a DataFrame with string columns, now called with an array of the same layout
        X = encode_frame(pd.DataFrame(np.asarray(X, dtype=object), columns=estimator._columns), categories).to_numpy()
    y_shape = _shape(y)
    if len(y_shape) == 2 and y_shape[1] > 1:
        wrapper = 'MultiOutputClassifier' if type(estimator).__name__.endswith('Classifier') else 'MultiOutputRegressor'
        raise ValueError(f'{type(estimator).__name__} predicts one target, but y has shape {y_shape}. For several targets, '
                         f'wrap it: sklearn.multioutput.{wrapper}({type(estimator).__name__}()).')
    try:
        return validate_data(estimator, X, y, reset=reset, allow_nan=allow_nan, dtype=[np.float64, np.float32], **check_params)
    except ValueError as error:
        if 'convert' in str(error) and 'string' in str(error):
            raise ValueError(f'{error}. {_STRING_HINT}') from error
        if 'y contains NaN' in str(error):
            raise ValueError(f'{error} Drop the rows whose target is missing first, e.g. keep = y.notna(); X, y = X[keep], y[keep].') from error
        raise


def check_sample_weight(sample_weight, n_samples):
    """Return the weights as a float64 array of shape (n_samples,), or None for unweighted data."""
    if sample_weight is None:
        return None
    if isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.ndim != 1 or sample_weight.shape[0] != n_samples:
        raise ValueError(f'sample_weight.shape == {sample_weight.shape}, expected {(n_samples,)}!')
    if not np.all(np.isfinite(sample_weight)):
        raise ValueError('sample_weight contains NaN or infinity.')
    if np.any(sample_weight < 0):
        raise ValueError('Negative values in data passed to `sample_weight`.')
    if not np.any(sample_weight > 0):
        raise ValueError('sample_weight is zero everywhere: at least one weight must be positive.')
    return sample_weight
