"""Compare defaults-study configurations: accuracy vs. its base learner and resource use.

    python compare_configs.py results/study_*.jsonl
"""
import json
import sys

import numpy as np
import pandas as pd

rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = 'macro_f1' if r.get('task') == 'classification' else 'r2'
            rows.append({'config': r.get('config'), 'dataset': r['dataset'], 'task': r.get('task'), 'status': r['status'],
                         'pdl': (r.get('pdl') or {}).get(key, np.nan), 'base': (r.get('base') or {}).get(key, np.nan),
                         'fit_s': r.get('pdl_fit_seconds', np.nan), 'predict_s': r.get('pdl_predict_seconds', np.nan),
                         'peak_gb': r.get('pdl_peak_gb', np.nan), 'nodes_M': (r.get('pdl_tree_nodes') or np.nan) / 1e6})
table = pd.DataFrame(rows)
common = table[table.status == 'ok'].groupby('dataset').config.nunique()
datasets = common[common == table.config.nunique()].index  # datasets where every config ran
ok = table[table.dataset.isin(datasets)].assign(gain=lambda d: d.pdl - d.base)
print(f'{len(datasets)} datasets where every configuration ran; failures per config:')
print(table[table.status != 'ok'].groupby(['config', 'status']).dataset.apply(list).to_string(), '\n')
summary = ok.groupby('config').agg(mean_pdl=('pdl', 'mean'), mean_gain=('gain', 'mean'), wins=('gain', lambda g: int((g > 0.005).sum())),
                                    losses=('gain', lambda g: int((g < -0.005).sum())), fit_median=('fit_s', 'median'),
                                    fit_max=('fit_s', 'max'), predict_median=('predict_s', 'median'), peak_median=('peak_gb', 'median'),
                                    peak_max=('peak_gb', 'max'), nodes_M_median=('nodes_M', 'median'))
print(summary.sort_values('mean_pdl', ascending=False).round(3).to_string())
print('\nper dataset (PDL score by config):')
print(ok.pivot(index='dataset', columns='config', values='pdl').round(3).to_string())
