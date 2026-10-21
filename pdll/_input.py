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
_TEMPORAL_INFERRED_TYPES = ('date', 'datetime', 'datetime64', 'time', 'timedelta', 'timedelta64', 'period')
_CONVERT_HINT = 'Convert it first, e.g. extract the year/month/day or use .astype("int64").'
_MISSING_TARGET_HINT = 'Drop the rows whose target is missing first, e.g. keep = y.notna(); X, y = X[keep], y[keep].'
_STRING_HINT = ('PDL needs numeric features. Pass a pandas DataFrame (pdll encodes string and categorical columns '
                'itself) or encode them first, e.g. with sklearn.preprocessing.OrdinalEncoder.')


def _is_plain_numeric(dtype) -> bool:
    """numpy int, uint and float dtypes (no bool, no pandas extension dtypes)."""
    return isinstance(dtype, np.dtype) and dtype.kind in 'iuf'


def _check_supported_dtype(column, dtype):
    if (pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype)
            or isinstance(dtype, (pd.PeriodDtype, pd.IntervalDtype))):
        raise TypeError(f'Column {column!r} has dtype {dtype}, but PDL needs numbers. {_CONVERT_HINT}')
    if pd.api.types.is_complex_dtype(dtype):
        raise TypeError(f'Column {column!r} holds complex numbers, which scikit-learn estimators do not support. Split it '
                        f'into two columns first, e.g. X[{column!r} + "_re"], X[{column!r} + "_im"] = X[{column!r}].real, X[{column!r}].imag.')


def _numbers_as_text(values: pd.Series):
    """Share of the distinct values of a string column that parse as numbers, and some that do not."""
    distinct = pd.Series(pd.unique(values.to_numpy(dtype=object))).astype(str).str.strip()
    parsed = pd.to_numeric(distinct, errors='coerce')
    return (parsed.notna().mean() if len(distinct) else 0.), list(distinct[parsed.isna()][:3])


def learn_categories(X: pd.DataFrame) -> dict:
    """Find the columns of X that must be encoded and fix their categories (position = code)."""
    duplicated = list(X.columns[X.columns.duplicated()].unique())
    if duplicated:
        raise ValueError(f'Column names must be unique, but {duplicated} appear more than once.')
    categories, id_like = {}, []
    for column in X.columns:
        series = X[column]
        dtype = series.dtype
        _check_supported_dtype(column, dtype)
        if isinstance(dtype, pd.CategoricalDtype):
            categories[column] = list(dtype.categories)  # keeps the order of ordered categoricals
            continue
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            continue
        inferred = pd.api.types.infer_dtype(series, skipna=True)
        if inferred in _NUMERIC_INFERRED_TYPES:
            continue  # object column holding numbers
        if inferred in _TEMPORAL_INFERRED_TYPES:  # e.g. datetime.date objects, as pyarrow date32 columns arrive
            raise TypeError(f'Column {column!r} holds {inferred} values, but PDL needs numbers. {_CONVERT_HINT}')
        if inferred.startswith('complex'):
            _check_supported_dtype(column, np.dtype(complex))
        values = series.dropna()
        try:
            categories[column] = sorted(pd.unique(values.to_numpy(dtype=object)), key=str)
        except TypeError as error:  # lists, dicts or arrays in the cells
            raise TypeError(f'Column {column!r} holds non-scalar values ({error}). PDL needs one number or category per '
                            f'cell: expand or drop the column first.') from error
        numeric_share, not_numbers = _numbers_as_text(values)
        if numeric_share == 1.:  # numbers stored as text, e.g. read_csv(dtype=str): convert them like scikit-learn does
            del categories[column]
            continue
        if numeric_share >= 0.9:
            warnings.warn(f'Column {column!r} holds numbers stored as text, plus values such as {not_numbers}. PDL encodes it '
                          f'as categories, which loses the order of the numbers; convert it first, e.g. '
                          f'pd.to_numeric(X[{column!r}], errors="coerce").', UserWarning, stacklevel=4)
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
                if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:  # numbers stored as text
                    arrays.append(pd.to_numeric(series.astype(object).where(series.notna(), np.nan).map(
                        lambda v: v.strip() if isinstance(v, str) else v)).to_numpy(dtype=np.float64, na_value=np.nan))
                else:
                    arrays.append(series.to_numpy(dtype=np.float64, na_value=np.nan))
            except (TypeError, ValueError) as error:
                raise ValueError(f'Column {column!r} could not be converted to numbers ({error}). {_STRING_HINT}') from error
    dtype = np.float32 if arrays and all(a.dtype == np.float32 for a in arrays) else np.float64
    values = np.column_stack([a.astype(dtype, copy=False) for a in arrays]) if arrays else np.empty((len(X), 0), dtype)
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def needs_encoding(X: pd.DataFrame) -> bool:
    return not all(_is_plain_numeric(dtype) for dtype in X.dtypes)


def _polars_to_pandas_without_pyarrow(data):
    """polars' own column-by-column conversion, for environments without pyarrow (which ``to_pandas`` needs)."""
    if hasattr(data, 'get_column'):
        return pd.DataFrame({name: data.get_column(name).to_numpy() for name in data.columns})
    return pd.Series(data.to_numpy(), name=data.name or None)


def as_pandas(data):
    """polars and pyarrow tables, series and arrays become pandas; everything else is returned unchanged.

    A polars LazyFrame is collected. pyarrow dates become datetime64 (rejected with advice, like pandas datetimes)
    instead of ``datetime.date`` objects. Anything that does not convert to a pandas DataFrame or Series is
    returned unchanged, for scikit-learn's own validation to handle.
    """
    library = type(data).__module__.split('.')[0]
    if library not in ('polars', 'pyarrow'):
        return data
    if library == 'polars' and type(data).__name__ == 'LazyFrame':
        data = data.collect()
    if not callable(getattr(data, 'to_pandas', None)):
        return data
    try:
        converted = data.to_pandas(date_as_object=False) if library == 'pyarrow' else data.to_pandas()
    except ImportError:  # polars without pyarrow
        converted = _polars_to_pandas_without_pyarrow(data) if library == 'polars' else data
    return converted if isinstance(converted, (pd.DataFrame, pd.Series)) else data


def _shape(y):
    """Shape of y without calling numpy functions on it (array-likes may forbid them); () when absent."""
    if y is None or (isinstance(y, str) and y == 'no_validation'):
        return ()
    shape = getattr(y, 'shape', None)
    return tuple(shape) if shape is not None else np.asarray(y).shape


def _check_same_columns(fitted, columns):
    """Encoding works by column name: report renamed, extra or missing columns the way scikit-learn does."""
    if fitted is None:
        return
    unseen = [c for c in columns if c not in fitted]
    missing = [c for c in fitted if c not in columns]
    if unseen or missing:
        message = 'The feature names should match those that were passed during fit.\n'
        if unseen:
            message += 'Feature names unseen at fit time:\n' + ''.join(f'- {c}\n' for c in unseen)
        if missing:
            message += 'Feature names seen at fit time, yet now missing:\n' + ''.join(f'- {c}\n' for c in missing)
        raise ValueError(message)


def _is_text_target(estimator, y):
    if type(estimator).__name__.endswith('Classifier') or not _shape(y):
        return False
    try:
        np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return True
    return False


def to_float_array(estimator, X, y='no_validation', *, reset, allow_nan, **check_params):
    """Validate X (and y) like scikit-learn does and return X as a float32/float64 ndarray.

    On ``reset`` (fit) the categorical columns of a DataFrame are learned and stored in ``estimator._categories``;
    afterwards the same encoding is applied, by column name, or by position if X is no longer a DataFrame.
    polars and pyarrow tables are handled as pandas DataFrames.
    """
    X, y = as_pandas(X), as_pandas(y)
    if reset:
        estimator._categories = learn_categories(X) if isinstance(X, pd.DataFrame) else {}
        estimator._columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
    categories = estimator._categories
    if isinstance(X, pd.DataFrame):
        if categories and not reset:
            _check_same_columns(estimator._columns, X.columns)
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
    if isinstance(y, pd.Series) and y.isna().any():  # string, object or nullable targets bypass scikit-learn's y message
        raise ValueError(f'Input y contains NaN. {_MISSING_TARGET_HINT}')
    try:
        return validate_data(estimator, X, y, reset=reset, allow_nan=allow_nan, dtype=[np.float64, np.float32], **check_params)
    except ValueError as error:
        if 'convert' in str(error) and 'string' in str(error):
            if _is_text_target(estimator, y):
                raise ValueError(f'{error}. {type(estimator).__name__} needs a numeric target; for class labels use '
                                 f'PairwiseDifferenceClassifier.') from error
            raise ValueError(f'{error}. {_STRING_HINT}') from error
        if 'y contains NaN' in str(error):
            raise ValueError(f'{error} {_MISSING_TARGET_HINT}') from error
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
