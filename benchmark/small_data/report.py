"""Markdown report of the small-data benchmark: ranks, head-to-head counts and the per-task table.

    python report.py results/cc18_small.jsonl [results/openml_public.json] > RESULTS.md

Ranks are per task (1 = best accuracy, ties share the mean rank) over the tasks where every model finished.
"Ties" in head-to-head counts are accuracy differences of at most 0.005.
"""
import json
import sys

import numpy as np
import pandas as pd

from models import MODELS

TIE = 0.005
LABELS = {'pdl': 'PDL (pdll 1.0)', 'logistic_regression': 'Logistic regression', 'knn': 'k-NN', 'svm': 'SVM (RBF)',
          'decision_tree': 'Decision tree', 'random_forest': 'Random forest', 'extra_trees': 'Extra trees',
          'hist_gradient_boosting': 'HistGradientBoosting', 'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'catboost': 'CatBoost',
          'tabpfn': 'TabPFN'}


def load(path):
    rows = [json.loads(line) for line in open(path)]
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(['task', 'model'], keep='last')


def public_rank(public, task, score):
    entry = public.get(str(task))
    if entry is None:
        return None, None
    values = list(entry['best_per_method'].values())
    return 1 + sum(v > score for v in values), max(values) if values else None


def summary_table(acc, frame, tasks):
    ranks = acc.loc[tasks].rank(axis=1, ascending=False, method='average')
    ok = frame[frame.status == 'ok'].set_index(['task', 'model'])
    rows = []
    for model in acc.columns:
        fit = ok.xs(model, level='model')['fit_seconds'] if model in ok.index.get_level_values('model') else pd.Series(dtype=float)
        rows.append({'model': LABELS.get(model, model), 'mean rank': ranks[model].mean(), 'mean accuracy': acc.loc[tasks, model].mean(),
                     'mean ROC AUC': ok.xs(model, level='model').reindex(tasks)['roc_auc'].mean(),
                     'top-3 finishes': int((ranks[model] <= 3).sum()), 'median fit s (10 folds)': fit.median()})
    table = pd.DataFrame(rows).sort_values('mean rank')
    return table.to_markdown(index=False, floatfmt='.3f')


def head_to_head(acc, tasks, model='pdl'):
    rows = []
    for other in acc.columns:
        if other == model:
            continue
        diff = acc.loc[tasks, model] - acc.loc[tasks, other]
        rows.append({'PDL vs': LABELS.get(other, other), 'wins': int((diff > TIE).sum()), 'ties': int((diff.abs() <= TIE).sum()),
                     'losses': int((diff < -TIE).sum()), 'mean difference': diff.mean()})
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt='+.3f')


def main():
    frame = load(sys.argv[1])
    public = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    models = [m for m in MODELS + ['tabpfn'] if m in set(frame.model)]
    acc = frame[frame.status == 'ok'].pivot(index='task', columns='model', values='accuracy').reindex(columns=models)
    info = frame.dropna(subset=['dataset']).groupby('task')[['dataset', 'rows']].first()
    complete = acc.dropna().index
    small = [t for t in complete if info.loc[t, 'rows'] <= 1000]
    larger = [t for t in complete if info.loc[t, 'rows'] > 1000]

    print(f'# Small-data benchmark: {len(complete)} OpenML-CC18 tasks, {len(models)} models\n')
    failures = frame[frame.status != 'ok']
    if len(failures):
        print('Runs that did not finish:\n')
        print(failures[['task', 'model', 'status', 'error']].to_markdown(index=False), '\n')
    for title, tasks in [('All tasks', list(complete)), ('Up to 1,000 rows', small), ('1,000 to 2,000 rows', larger)]:
        if not tasks:
            continue
        print(f'## {title} ({len(tasks)} tasks)\n')
        print(summary_table(acc, frame, tasks), '\n')
        print(head_to_head(acc, tasks), '\n')

    print('## Accuracy per task (10-fold, official OpenML split)\n')
    table = acc.copy()
    table.insert(0, 'rows', info['rows'])
    table.insert(0, 'dataset', info['dataset'])
    if public:
        ranks = [public_rank(public, t, acc.loc[t, 'pdl']) for t in table.index]
        table['PDL rank among public OpenML methods'] = [r for r, _ in ranks]
        table['best public'] = [b for _, b in ranks]
    table = table.sort_values('rows').rename(columns=LABELS)
    print(table.to_markdown(floatfmt='.3f'))


if __name__ == '__main__':
    main()
