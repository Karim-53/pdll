"""scikit-learn's own estimator checks (API conventions, input validation, cloning, pickling, ...)."""
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

# Checks that cannot pass by design, with the reason.
EXPECTED_FAILURES = {
    # Integer weights vs. repeated rows: equivalent in PDL's maths, but random tree ensembles draw different splits
    # for different row counts (scikit-learn marks its own forests the same way).
    'check_sample_weight_equivalence_on_dense_data': 'randomized base estimator',
    'check_sample_weight_equivalence_on_sparse_data': 'randomized base estimator',
    'check_sample_weights_invariance': 'randomized base estimator',
}


def _estimators():
    return [
        # small trees: these checks test the API, not accuracy
        PairwiseDifferenceClassifier(ExtraTreesClassifier(n_estimators=10, min_samples_leaf=3, class_weight='balanced', n_jobs=-1), random_state=0),
        PairwiseDifferenceRegressor(ExtraTreesRegressor(n_estimators=10, min_samples_leaf=5, n_jobs=-1), random_state=0),
    ]


@parametrize_with_checks(_estimators())
def test_sklearn_compatible_estimator(estimator, check):
    name = getattr(check, 'func', check).__name__
    if name in EXPECTED_FAILURES:
        pytest.xfail(EXPECTED_FAILURES[name])
    check(estimator)
