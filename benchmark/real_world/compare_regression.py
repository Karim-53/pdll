"""Regression part of the defaults study: R² and resource use per configuration."""
import glob
import json

import pandas as pd

rows = []
for path in glob.glob('results/study_*.jsonl'):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('task') == 'regression' and r['status'] == 'ok':
                rows.append({'config': r['config'], 'dataset': r['dataset'], 'r2': r['pdl']['r2'], 'base': r['base']['r2'],
                             'fit': r['pdl_fit_seconds'], 'pred': r['pdl_predict_seconds'], 'peak': r['pdl_peak_gb'],
                             'nodesM': (r.get('pdl_tree_nodes') or 0) / 1e6})
d = pd.DataFrame(rows)
print(d.pivot(index='dataset', columns='config', values='r2').round(3).to_string())
print(d.groupby('config').agg(r2=('r2', 'mean'), base=('base', 'mean'), fit_med=('fit', 'median'), fit_max=('fit', 'max'),
                              peak_max=('peak', 'max'), nodesM=('nodesM', 'median')).round(3).to_string())
