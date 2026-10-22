"""Summarise a campaign's robustness (did each dataset run, how long, how much memory) as Markdown.

    python report.py results/campaign_v1.0.0.jsonl > RESULTS.md
"""
import json
import sys

import pandas as pd

EXPECTED_REFUSALS = {'sklearn_20newsgroups_vectorized': 'sparse input', 'sklearn_linnerud': 'multi-output target'}


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main(paths):
    rows = load(paths)
    table = pd.DataFrame([{
        'dataset': r['dataset'], 'source': r.get('source'), 'task': r.get('task'), 'status': r['status'],
        'rows': (r.get('shape') or [None])[0], 'cols': (r.get('shape') or [None, None])[-1],
        'fit_s': r.get('pdl_fit_seconds'), 'predict_s': r.get('pdl_predict_seconds'), 'peak_gb': r.get('pdl_peak_gb'),
        'error': (r.get('error') or '')[:120],
    } for r in rows])
    ok = table[table.status == 'ok']
    refused = table[(table.status != 'ok') & table.dataset.isin(EXPECTED_REFUSALS)]
    failed = table[(table.status != 'ok') & ~table.dataset.isin(EXPECTED_REFUSALS)]
    print(f'# pdll real-world validation\n\n{len(table)} datasets: {len(ok)} ran, {len(refused)} refused as expected '
          f'({", ".join(sorted(EXPECTED_REFUSALS.values()))}), {len(failed)} failed.\n')
    for task in ('classification', 'regression'):
        part = ok[ok.task == task]
        if len(part):
            print(f'**{task}** ({len(part)} datasets): median fit {part.fit_s.median():.0f}s, median peak memory '
                  f'{part.peak_gb.median():.1f} GB, max {part.peak_gb.max():.1f} GB.\n')
    if len(failed):
        print('## Failures\n')
        print(failed[['dataset', 'source', 'status', 'rows', 'cols', 'error']].to_markdown(index=False))
        print()
    print('## All datasets\n')
    print(table.drop(columns='error').round(3).to_markdown(index=False))


if __name__ == '__main__':
    main(sys.argv[1:])
