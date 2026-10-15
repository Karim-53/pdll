"""pdll 1.x must reproduce pdll 0.3.1 (the code behind the paper) on the inputs the paper used.

The reference outputs in ``golden/v0_3_1.npz`` were produced by ``golden/make_golden_v031.py``.
Cases with a fake (deterministic) base estimator are checked on every scikit-learn version. Cases with real
trees are checked only on the scikit-learn release that produced the file, since tree building may change
between releases.
"""
import json
import os
import sys

import numpy as np
import pytest
import scipy
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'golden'))

from golden_cases import SKLEARN_DEPENDENT, classification_cases, regression_cases  # noqa: E402
from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor  # noqa: E402

GOLDEN = np.load(os.path.join(HERE, 'golden', 'v0_3_1.npz'))
with open(os.path.join(HERE, 'golden', 'v0_3_1.json')) as f:
    META = json.load(f)


def _skip_if_other_sklearn(name):
    if name.split('__')[1] in SKLEARN_DEPENDENT and sklearn.__version__ != META['sklearn']:
        pytest.skip(f"tree outputs recorded with scikit-learn {META['sklearn']}, running {sklearn.__version__}")


@pytest.mark.parametrize('name, data, make_estimator', list(classification_cases()), ids=lambda v: v if isinstance(v, str) else '')
def test_classifier_matches_0_3_1(name, data, make_estimator):
    _skip_if_other_sklearn(name)
    X_train, y_train, X_test = data
    pdc = PairwiseDifferenceClassifier(estimator=make_estimator()).fit(X_train, y_train)
    assert pdc.decide_use_prior() == bool(GOLDEN[f'{name}__use_prior'])
    np.testing.assert_allclose(pdc.predict_proba(X_test), GOLDEN[f'{name}__proba'], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('name, data, make_estimator, weighting', list(regression_cases()), ids=lambda v: v if isinstance(v, str) else '')
def test_regressor_matches_0_3_1(name, data, make_estimator, weighting):
    _skip_if_other_sklearn(name)
    X_train, y_train, X_test, X_val, y_val = data
    pdr = PairwiseDifferenceRegressor(estimator=make_estimator()).fit(X_train, y_train)
    np.testing.assert_allclose(pdr.predict(X_test), GOLDEN[f'{name}__pred'], rtol=1e-10, atol=1e-10)
    if weighting is not None:
        if (np.__version__, scipy.__version__) != (META['numpy'], META['scipy']):
            return  # SLSQP amplifies 1-ulp differences from other BLAS / optimizer builds: compare on the recorded stack only
        pdr.learn_anchor_weights(X_val, y_val, method=weighting, enable_warnings=False)
        # 0.3.1's predict() overwrote zero anchor weights with NaN in place (a bug fixed in 1.0): NaN there means 0.
        expected_weights = np.nan_to_num(GOLDEN[f'{name}__weights'], nan=0.)
        np.testing.assert_allclose(pdr.sample_weight_.to_numpy(), expected_weights, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(pdr.predict(X_test), GOLDEN[f'{name}__pred_weighted'], rtol=1e-8, atol=1e-8)
