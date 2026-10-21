"""Statistics of the small-data benchmark, following the plan declared before the results were read.

    pip install autorank==1.3.0 baycomp==1.0.3
    python analyze.py results/cc18_small.jsonl --figures figures/ > ANALYSIS.md

- Every one of the 35 tasks stays in: a model without a result on a task gets the accuracy of always predicting the
  majority class there (dropping such tasks would reward models that fail on hard tasks).
- Friedman test and Nemenyi critical difference over all models (autorank, non-parametric mode), drawn as a CD diagram.
- PDL against each competitor: wins/ties/losses (tie = 0.5 pp), Wilcoxon signed-rank (zero_method='zsplit') with Holm
  correction over these 10 comparisons, and the Bayesian signed-rank test with a +-1 pp region of practical equivalence.
- Each verdict is worded by the declared table; the outcome (A top-3, B middle, C bottom) follows the declared rule.
- Sensitivity: without the tasks that were used to choose pdll's defaults, and with the six mfeat views averaged.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from models import MODELS
from report import LABELS, load
from run_all import CC18_SMALL

TIE, ROPE, ALPHA, CONTROL = 0.005, 0.01, 0.05, 'pdl'
# tasks of the 20-dataset defaults study (max_pairs, max_anchors) and of imbalance_probe.py / the E2E gate
TUNED = {'banknote-authentication', 'car', 'cmc', 'credit-g', 'mfeat-morphological', 'pc1', 'qsar-biodeg',
         'steel-plates-fault', 'vehicle', 'ilpd', 'pc4', 'diabetes'}


def majority_accuracy(task_id):
    import openml
    _, y = openml.tasks.get_task(task_id).get_X_and_y(dataset_format='dataframe')
    return float(y.value_counts(normalize=True).max())


def accuracy_matrix(frame, models):
    acc = frame[frame.status == 'ok'].pivot(index='task', columns='model', values='accuracy').reindex(index=CC18_SMALL, columns=models)
    imputed = acc.isna()
    for task in acc.index[imputed.any(axis=1)]:
        acc.loc[task] = acc.loc[task].fillna(majority_accuracy(task))
    return acc, imputed


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1., running)
    return adjusted


def verdict(row, other):
    if row['holm_p'] < ALPHA and row['mean_diff'] > 0:
        return f'more accurate than {other} on these tasks'
    if row['holm_p'] < ALPHA and row['mean_diff'] < 0:
        return f'{other} is more accurate than PDL'
    if row['p_rope'] > 0.95:
        return f'practically equivalent to {other} (within 1 pp)'
    if row['p_rope'] + row['p_pdl'] > 0.95:
        return f'not practically worse than {other}'
    return f'inconclusive (95% interval of the mean difference: {row["ci_low"]:+.3f} to {row["ci_high"]:+.3f})'


def pdl_comparisons(acc):
    import baycomp
    rows = []
    for other in [m for m in acc.columns if m != CONTROL]:
        d = (acc[CONTROL] - acc[other]).round(4)
        boot = np.random.default_rng(0).choice(d.to_numpy(), size=(10000, len(d))).mean(axis=1)
        p_pdl, p_rope, p_other = baycomp.SignedRankTest.probs(acc[CONTROL].to_numpy(), acc[other].to_numpy(), rope=ROPE)
        rows.append({'model': other, 'wins': int((d > TIE).sum()), 'ties': int((d.abs() <= TIE).sum()), 'losses': int((d < -TIE).sum()),
                     'mean_diff': d.mean(), 'ci_low': np.quantile(boot, 0.025), 'ci_high': np.quantile(boot, 0.975),
                     'wilcoxon_p': stats.wilcoxon(d, zero_method='zsplit').pvalue if (d != 0).any() else 1.,
                     'p_pdl': p_pdl, 'p_rope': p_rope, 'p_other': p_other})
    table = pd.DataFrame(rows)
    table['holm_p'] = holm(table['wilcoxon_p'].to_numpy())
    table['verdict'] = [verdict(row, LABELS.get(row['model'], row['model'])) for _, row in table.iterrows()]
    return table


def outcome(ranks, comparisons, top_group):
    pdl_place = int(ranks.rank()[CONTROL])
    worse = comparisons[(comparisons.holm_p < ALPHA) & (comparisons.mean_diff < 0)]
    if pdl_place >= 9 or len(worse) >= 5 or 'extra_trees' in set(worse.model):
        return 'C (bottom)', pdl_place
    if pdl_place <= 3 and CONTROL in top_group and worse.empty:
        return 'A (top-3)', pdl_place
    return 'B (middle)', pdl_place


def rank_summary(acc):
    return acc.rank(axis=1, ascending=False, method='average').mean().sort_values()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--figures', default='figures')
    args = parser.parse_args()
    from autorank import autorank, plot_stats

    frame = load(args.results)
    models = [m for m in MODELS if m in set(frame.model)]
    acc, imputed = accuracy_matrix(frame, models)
    names = frame.dropna(subset=['dataset']).groupby('task')['dataset'].first()

    print('# Small-data benchmark: statistics\n')
    print(f'{len(acc)} tasks x {len(models)} models. Imputed (no result, majority-class accuracy): '
          f'{int(imputed.to_numpy().sum())} of {imputed.size}: '
          f'{ {LABELS.get(m, m): int(imputed[m].sum()) for m in models if imputed[m].any()} }\n')

    result = autorank(acc.rename(columns=LABELS), alpha=ALPHA, order='descending', force_mode='nonparametric', verbose=False)
    ranks = rank_summary(acc)
    best = ranks.index[0]
    top_group = [m for m in models if ranks[m] - ranks[best] <= result.cd]
    print(f'Friedman p = {result.pvalue:.2g}; Nemenyi critical difference = {result.cd:.2f} ranks; '
          f'top group (within the CD of {LABELS.get(best, best)}): {[LABELS.get(m, m) for m in top_group]}\n')
    print(pd.DataFrame({'mean rank': ranks, 'mean accuracy': acc.mean()[ranks.index]}).rename(index=LABELS)
          .to_markdown(floatfmt='.3f'), '\n')

    comparisons = pdl_comparisons(acc)
    shown = comparisons.assign(model=comparisons.model.map(lambda m: LABELS.get(m, m)))
    print('## PDL against each model\n')
    print(shown[['model', 'wins', 'ties', 'losses', 'mean_diff', 'ci_low', 'ci_high', 'wilcoxon_p', 'holm_p', 'p_pdl', 'p_rope',
                 'p_other', 'verdict']].to_markdown(index=False, floatfmt='.3f'), '\n')
    label, place = outcome(ranks, comparisons, top_group)
    print(f'**Declared outcome: {label}** (PDL mean-rank place {place} of {len(models)}).\n')

    print('## Sensitivity\n')
    names_by_task = names.reindex(acc.index)
    untuned = acc[~names_by_task.isin(TUNED).to_numpy()]
    mfeat = names_by_task.str.startswith('mfeat').to_numpy()
    collapsed = pd.concat([acc[~mfeat], acc[mfeat].mean().to_frame('mfeat (6 views averaged)').T])
    for title, data in [(f'Without the {len(acc) - len(untuned)} tasks used to choose the defaults', untuned),
                        ('mfeat views averaged into one task', collapsed)]:
        r = rank_summary(data)
        et = pdl_comparisons(data).set_index('model').loc['extra_trees']
        print(f'- {title} ({len(data)} tasks): PDL mean rank {r[CONTROL]:.2f} (place {int(r.rank()[CONTROL])}); '
              f'against ExtraTrees: {et.wins}/{et.ties}/{et.losses}, Holm p {et.holm_p:.3f}, {et.verdict}')

    os.makedirs(args.figures, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_stats(result, allow_insignificant=True)
    plt.savefig(os.path.join(args.figures, 'cd_diagram.png'), dpi=200, bbox_inches='tight')
    with open(os.path.join(args.figures, 'comparisons.json'), 'w') as f:
        json.dump(comparisons.to_dict(orient='records'), f, indent=1)


if __name__ == '__main__':
    main()
