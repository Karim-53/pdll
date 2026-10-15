"""Record the outputs of pdll 0.3.1 so that later versions can prove they reproduce the paper.

This file documents how ``v0_3_1.npz`` was produced. It needs the 0.3.1 source code, i.e. commit
f781c1a (identical to the 0.3.1 release on PyPI):

    git worktree add ../pdll-0.3.1 f781c1a
    python tests/golden/make_golden_v031.py ../pdll-0.3.1

Only numpy inputs are recorded: that is what the paper benchmark used (``run_benchmark.py`` passes
``X.values`` / ``y.values``), and it is the input type on which 0.3.1 behaves correctly.
"""
import json
import os
import sys

import numpy as np
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tests/ for helpers


def main(old_source_dir):
    sys.path.insert(0, os.path.abspath(old_source_dir))
    import pdll
    assert pdll.__version__ == '0.3.1', pdll.__version__
    from golden_cases import classification_cases, regression_cases
    from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

    arrays, meta = {}, {'pdll': pdll.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'cases': []}
    for name, (X_train, y_train, X_test), make_estimator in classification_cases():
        pdc = PairwiseDifferenceClassifier(estimator=make_estimator()).fit(X_train, y_train)
        arrays[f'{name}__proba'] = np.asarray(pdc.predict_proba(X_test))
        arrays[f'{name}__use_prior'] = np.asarray(pdc.decide_use_prior())
        meta['cases'].append(name)
    for name, (X_train, y_train, X_test, X_val, y_val), make_estimator, weighting in regression_cases():
        pdr = PairwiseDifferenceRegressor(estimator=make_estimator()).fit(X_train, y_train)
        arrays[f'{name}__pred'] = np.asarray(pdr.predict(X_test))
        if weighting is not None:
            pdr.learn_anchor_weights(X_val, y_val, method=weighting, enable_warnings=False)
            arrays[f'{name}__pred_weighted'] = np.asarray(pdr.predict(X_test))
            arrays[f'{name}__weights'] = np.asarray(pdr.sample_weight_)
        meta['cases'].append(name)

    np.savez_compressed(os.path.join(HERE, 'v0_3_1.npz'), **arrays)
    with open(os.path.join(HERE, 'v0_3_1.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('saved', len(arrays), 'arrays;', meta)


if __name__ == '__main__':
    main(sys.argv[1])
