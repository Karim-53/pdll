"""Regenerate ``baseline.json`` for the performance regression gate (a deliberate step, commit the result)."""
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn

import pdll
from benchmark import BASELINE_PATH, regression_summary, run, run_regression, summary

if __name__ == '__main__':
    results, regression_results = run(), run_regression()
    baseline = {
        'environment': {'pdll': pdll.__version__, 'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit-learn': sklearn.__version__, 'scipy': scipy.__version__,
                        'machine': platform.machine(), 'system': platform.system()},
        'summary': summary(results),
        'results': results,
        'regression': {'summary': regression_summary(regression_results), 'results': regression_results},
    }
    with open(BASELINE_PATH, 'w') as f:
        json.dump(baseline, f, indent=1)
    print(json.dumps(baseline['summary'], indent=1))
    print(json.dumps(baseline['regression']['summary'], indent=1))
