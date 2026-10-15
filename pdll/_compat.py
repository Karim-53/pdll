"""Bridges between the scikit-learn versions supported by pdll (>= 1.2)."""
import inspect

from sklearn.utils.validation import check_array

# scikit-learn 1.6 renamed check_array(force_all_finite=...) to ensure_all_finite
FINITE_KEYWORD = 'ensure_all_finite' if 'ensure_all_finite' in inspect.signature(check_array).parameters else 'force_all_finite'

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
except ImportError:  # pragma: no cover - exercised on old scikit-learn only
    _sklearn_validate_data = None

try:  # scikit-learn >= 1.6
    from sklearn.utils import get_tags as _get_tags
except ImportError:  # pragma: no cover - exercised on old scikit-learn only
    _get_tags = None


# scikit-learn >= 1.6 reads tags from __sklearn_tags__, older versions from _more_tags
NEW_TAGS_API = _get_tags is not None


def validate_data(estimator, X, y='no_validation', *, reset, allow_nan, **check_params):
    """Validate X (and y), and set or check ``n_features_in_`` / ``feature_names_in_``."""
    check_params[FINITE_KEYWORD] = 'allow-nan' if allow_nan else True
    if _sklearn_validate_data is not None:
        return _sklearn_validate_data(estimator, X, y, reset=reset, **check_params)
    return estimator._validate_data(X, y, reset=reset, **check_params)  # pragma: no cover


def allows_nan(estimator) -> bool:
    """Whether ``estimator`` accepts NaN in X, according to its scikit-learn tags.

    Estimators whose tags cannot be read (not scikit-learn based) are trusted to validate their own input.
    """
    try:
        if _get_tags is not None:
            return bool(_get_tags(estimator).input_tags.allow_nan)
        from sklearn.utils._tags import _safe_tags  # pragma: no cover
        return bool(_safe_tags(estimator, key='allow_nan'))  # pragma: no cover
    except Exception:  # noqa: BLE001 - any failure means "unknown"
        return True
