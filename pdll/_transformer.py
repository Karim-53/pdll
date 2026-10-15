"""PDCDataTransformer: the preprocessing of the paper benchmark, as a reusable transformer."""
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

_Y_TYPES = ('numeric', 'ordinal', 'string')


class PDCDataTransformer(TransformerMixin, BaseEstimator):
    """Scale numeric columns, one-hot encode string columns and ordinal-encode ordinal columns.

    Since pdll 1.0, :class:`PairwiseDifferenceClassifier` and :class:`PairwiseDifferenceRegressor` accept
    DataFrames with string or categorical columns directly. This transformer remains for explicit pipelines
    and reproduces the preprocessing of the paper benchmark.

    Parameters
    ----------
    numeric_features, ordinal_features, string_features : list of column names, default=None
        If all three are None, columns are grouped by dtype: numbers are numeric, ordered categoricals are
        ordinal, everything else (strings, bools, unordered categoricals) is one-hot encoded.
    y_type : {'numeric', 'ordinal', 'string'} or None, default=None
        If set, ``fit`` also learns a transformation of y, which ``transform(X, y)`` applies.
    """

    def __init__(self, numeric_features: Iterable = None, ordinal_features: Iterable = None,
                 string_features: Iterable = None, y_type: str = None):
        self.numeric_features = numeric_features
        self.ordinal_features = ordinal_features
        self.string_features = string_features
        self.y_type = y_type

    @staticmethod
    def _group_columns(X: pd.DataFrame):
        numeric, ordinal, string = [], [], []
        for column in X.columns:
            dtype = X[column].dtype
            if pd.api.types.is_bool_dtype(dtype):
                string.append(column)
            elif isinstance(dtype, pd.CategoricalDtype):
                (ordinal if dtype.ordered else string).append(column)
            elif pd.api.types.is_numeric_dtype(dtype):
                numeric.append(column)
            else:
                string.append(column)
        return numeric, ordinal, string

    @staticmethod
    def _as_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _cast_numeric(self, X):
        X = X.copy()
        for column in self.numeric_features_:
            X[column] = X[column].to_numpy(dtype=np.float64, na_value=np.nan)  # uint, nullable ints -> float
        return X

    def fit(self, X, y=None):
        if self.y_type is not None and self.y_type not in _Y_TYPES:
            raise ValueError(f"y_type must be one of 'numeric', 'ordinal', 'string' but got {self.y_type}")
        X = self._as_frame(X)
        if self.numeric_features is None and self.ordinal_features is None and self.string_features is None:
            self.numeric_features_, self.ordinal_features_, self.string_features_ = self._group_columns(X)
        else:
            self.numeric_features_ = list(self.numeric_features or [])
            self.ordinal_features_ = list(self.ordinal_features or [])
            self.string_features_ = list(self.string_features or [])
        self.preprocessing_ = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features_),
                ('str', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.string_features_),
                ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.ordinal_features_),
            ],
            remainder='passthrough',
            sparse_threshold=0.,
        ).fit(self._cast_numeric(X))

        self.preprocessing_y_ = None
        if self.y_type == 'numeric':
            self.preprocessing_y_ = StandardScaler()
        elif self.y_type == 'ordinal':
            self.preprocessing_y_ = OrdinalEncoder()
        elif self.y_type == 'string':
            self.preprocessing_y_ = OneHotEncoder(sparse_output=False)
        if y is not None and self.preprocessing_y_ is not None:
            self.preprocessing_y_.fit(np.asarray(y).reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        """Transform X to a float32 array; with y (and a fitted ``y_type``) return ``(X, y)``."""
        check_is_fitted(self)
        X = self.preprocessing_.transform(self._cast_numeric(self._as_frame(X)))
        X = np.asarray(X, dtype=np.float32)
        if y is None:
            return X
        if self.preprocessing_y_ is not None:
            y = self.preprocessing_y_.transform(np.asarray(y).reshape(-1, 1))
        return X, np.asarray(y)
