"""pdll 1.x must reproduce pdll 0.3.1 (the code behind the paper) on the inputs the paper used.

The reference outputs in ``golden/v0_3_1.npz`` were produced by ``golden/make_golden_v031.py``.
Cases with a fake (deterministic) base estimator are checked on every scikit-learn version. Cases with real
trees are checked only on the scikit-learn release that produced the file, since tree building may change
between releases.
"""
import json
import os
import platform
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


IS_X86_64 = platform.machine().lower() in ('x86_64', 'amd64')


def _skip_if_other_sklearn(name):
    if name.split('__')[1] in SKLEARN_DEPENDENT:
        if sklearn.__version__ != META['sklearn']:
            pytest.skip(f"tree outputs recorded with scikit-learn {META['sklearn']}, running {sklearn.__version__}")
        if not IS_X86_64:  # e.g. Apple Silicon: float rounding (FMA) changes which split wins a near-tie
            pytest.skip(f'tree outputs recorded on {META["machine"]}, running on {platform.machine()}')


@pytest.mark.parametrize('name, data, make_estimator', list(classification_cases()), ids=lambda v: v if isinstance(v, str) else '')
def test_classifier_matches_0_3_1(name, data, make_estimator):
    _skip_if_other_sklearn(name)
    X_train, y_train, X_test = data
    # 0.3.1's aggregation, chosen explicitly: since 1.0, use_prior='auto' no longer picks the prior-based posterior for
    # imbalanced multiclass data (wine_prior: 3 rows in one class), where it never predicts the tiny class
    used_prior = bool(GOLDEN[f'{name}__use_prior'])
    pdc = PairwiseDifferenceClassifier(estimator=make_estimator(), use_prior=used_prior).fit(X_train, y_train)
    np.testing.assert_allclose(pdc.predict_proba(X_test), GOLDEN[f'{name}__proba'], rtol=1e-10, atol=1e-12)
    expected_auto = used_prior and not name.startswith('wine_prior')
    assert PairwiseDifferenceClassifier(estimator=make_estimator()).fit(X_train, y_train).decide_use_prior() == expected_auto


@pytest.mark.parametrize('name, data, make_estimator, weighting', list(regression_cases()), ids=lambda v: v if isinstance(v, str) else '')
def test_regressor_matches_0_3_1(name, data, make_estimator, weighting):
    _skip_if_other_sklearn(name)
    X_train, y_train, X_test, X_val, y_val = data
    pdr = PairwiseDifferenceRegressor(estimator=make_estimator()).fit(X_train, y_train)
    np.testing.assert_allclose(pdr.predict(X_test), GOLDEN[f'{name}__pred'], rtol=1e-10, atol=1e-10)
    if weighting is not None:
        if (np.__version__, scipy.__version__) != (META['numpy'], META['scipy']):
            return  # other optimizer builds may converge elsewhere
        pdr.learn_anchor_weights(X_val, y_val, method=weighting, enable_warnings=False)
        # SLSQP (numerical gradients of a non-smooth MAE) amplifies 1-ulp differences of the CPU's BLAS kernels into
        # visibly different weights, with 0.3.1 as well. Weights match 0.3.1 exactly on the recording machine only;
        # everywhere we check that the weighted predictions stay close.
        np.testing.assert_allclose(pdr.predict(X_test), GOLDEN[f'{name}__pred_weighted'], rtol=5e-2)
