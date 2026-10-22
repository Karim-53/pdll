"""Statistics and figures of the small-data benchmark, following the plan declared before the results were read.

    pip install autorank==1.3.0 baycomp==1.0.3
    python analyze.py results/cc18_small.jsonl --figures figures/ > ANALYSIS.md

- Every one of the 35 tasks stays in: a model without a result on a task gets the score of always predicting the
  majority class there (dropping such tasks would reward models that fail on hard tasks).
- Friedman test and Nemenyi critical difference over all models (autorank, non-parametric mode), drawn as a CD diagram.
- PDL against each competitor: wins/ties/losses (tie = 0.5 pp), Wilcoxon signed-rank (zero_method='zsplit') with Holm
  correction over these 10 comparisons, and the Bayesian signed-rank test with a +-1 pp region of practical equivalence.
- Each verdict is worded by the declared table; the outcome (A top-3, B middle, C bottom) follows the declared rule.
- Sensitivity: without the tasks used to choose pdll's defaults, with the six mfeat views averaged, ROC AUC and
  macro-F1 instead of accuracy, and tie thresholds of 0 and 1 pp.
- Figures: CD diagram (F1), PDL minus each model per task (F2), mean rank against fit time (F3), PDL minus the best
  other model against training rows (F4).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from models import MODELS
from report import LABELS, load
from run_all import CC18_SMALL
from scipy import stats

TIE, ROPE, ALPHA, CONTROL = 0.005, 0.01, 0.05, 'pdl'
SEED = 0  # baycomp samples its posterior: a fixed seed makes borderline verdicts reproducible
DEFAULTS_STUDY = {'banknote-authentication', 'car', 'cmc', 'credit-g', 'mfeat-morphological', 'pc1', 'qsar-biodeg',
                  'steel-plates-fault', 'vehicle'}  # the 20-dataset study that chose max_pairs and max_anchors
OTHER_TUNING = {'ilpd', 'pc4', 'diabetes'}  # imbalance_probe.py and the E2E gate
# figure colours: validated pair (dataviz palette, light surface) plus neutral ink
BLUE, RED, GRAY, INK, MUTED, SURFACE, GRID = '#2a78d6', '#e34948', '#8f8d88', '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e1'
LABEL_OFFSETS = {'logistic_regression': (-8, -14), 'svm': (6, 6), 'hist_gradient_boosting': (6, -13), 'pdl': (8, 5),
                 'extra_trees': (6, 5), 'random_forest': (6, 6), 'lightgbm': (-52, -3), 'xgboost': (6, -12)}  # F3 label placement, in points


def task_target(task_id):
    import openml
    _, y = openml.tasks.get_task(task_id).get_X_and_y(dataset_format='dataframe')
    return y


def constant_score(task_id, metric):
    """Score of always predicting the majority class: the value a missing result gets."""
    y = task_target(task_id)
    p = float(y.value_counts(normalize=True).max())
    if metric == 'accuracy':
        return p
    if metric == 'roc_auc':
        return 0.5
    return 2 * p / (1 + p) / y.nunique()  # macro-F1: F1 of the majority class, 0 for the others


def score_matrix(frame, models, metric='accuracy'):
    scores = frame[frame.status == 'ok'].pivot(index='task', columns='model', values=metric).reindex(index=CC18_SMALL, columns=models)
    imputed = scores.isna()
    for task in scores.index[imputed.any(axis=1)]:
        scores.loc[task] = scores.loc[task].fillna(constant_score(task, metric))
    return scores, imputed


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


def wtl(d, tie):
    return int((d > tie).sum()), int((d.abs() <= tie).sum()), int((d < -tie).sum())


def pdl_comparisons(scores, fit=None):
    import baycomp
    rows = []
    for other in [m for m in scores.columns if m != CONTROL]:
        d = (scores[CONTROL] - scores[other]).round(4)
        boot = np.random.default_rng(0).choice(d.to_numpy(), size=(10000, len(d))).mean(axis=1)
        p_pdl, p_rope, p_other = baycomp.SignedRankTest.probs(scores[CONTROL].to_numpy(), scores[other].to_numpy(), rope=ROPE,
                                                              random_state=SEED)
        wins, ties, losses = wtl(d, TIE)
        row = {'model': other, 'wins': wins, 'ties': ties, 'losses': losses, 'mean_diff': d.mean(),
               'ci_low': np.quantile(boot, 0.025), 'ci_high': np.quantile(boot, 0.975),
               'wilcoxon_p': stats.wilcoxon(d, zero_method='zsplit').pvalue if (d != 0).any() else 1.,
               'p_pdl': p_pdl, 'p_rope': p_rope, 'p_other': p_other}
        if fit is not None:
            row['fit_ratio'] = fit[CONTROL].median() / fit[other].median()
        rows.append(row)
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


def rank_summary(scores):
    return scores.rank(axis=1, ascending=False, method='average').mean().sort_values()


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_differences(scores, comparisons, path):
    """F2: PDL minus each model, one dot per task; blue = PDL better, red = worse, gray = within 0.5 pp."""
    import matplotlib.pyplot as plt
    order = comparisons.sort_values('mean_diff')['model'].tolist()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.axvspan(-ROPE * 100, ROPE * 100, color='#e3e1db', linewidth=0, zorder=0)
    ax.axvline(0, color=GRAY, linewidth=1)
    rng = np.random.default_rng(0)
    for i, model in enumerate(order):
        d = (scores[CONTROL] - scores[model]).round(4)
        colours = np.where(d > TIE, BLUE, np.where(d < -TIE, RED, GRAY))
        ax.scatter(d * 100, i + rng.uniform(-0.18, 0.18, len(d)), s=22, c=colours, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        row = comparisons.set_index('model').loc[model]
        ax.text(1.01, i, f'{row.wins}/{row.ties}/{row.losses}  Holm p {row.holm_p:.2f}', transform=ax.get_yaxis_transform(),
                va='center', fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(order)), [f'vs {LABELS.get(m, m)}' for m in order], color=INK)
    ax.set_xlabel('PDL accuracy minus the other model (percentage points), one dot per task; shaded: ±1 pp', color=MUTED, fontsize=9)
    ax.set_title('PDL against each default model on 35 OpenML-CC18 tasks (W/T/L at 0.5 pp)', color=INK, fontsize=11, loc='left')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def figure_rank_vs_time(ranks, fit, path):
    """F3: mean rank against median fit time (10 folds), PDL highlighted, the Pareto front drawn."""
    import matplotlib.pyplot as plt
    median_fit = fit.median()
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    _style(ax)
    points = sorted((median_fit[m], ranks[m], m) for m in ranks.index if not np.isnan(median_fit[m]))  # all-failed models have no time
    front, best = [], np.inf
    for t, r, m in points:
        if r < best:
            front.append((t, r))
            best = r
    ax.plot([t for t, _ in front], [r for _, r in front], color=GRAY, linewidth=1, linestyle='--', zorder=2)
    for t, r, m in points:
        colour = BLUE if m == CONTROL else GRAY
        ax.scatter(t, r, s=64 if m == CONTROL else 36, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        dx, dy = LABEL_OFFSETS.get(m, (6, 4))
        ax.annotate(LABELS.get(m, m), (t, r), xytext=(dx, dy), textcoords='offset points', fontsize=8.5,
                    color=INK if m == CONTROL else MUTED, fontweight='bold' if m == CONTROL else 'normal')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('median fit time per task, 10 folds (seconds, log scale)', color=MUTED, fontsize=9)
    ax.set_ylabel('mean rank by accuracy (1 = best)', color=MUTED, fontsize=9)
    ax.set_title('Accuracy rank against training cost (dashed: Pareto front)', color=INK, fontsize=11, loc='left')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def figure_gap_vs_rows(scores, rows, path):
    """F4: PDL minus the best other model on each task, against the number of training rows."""
    import matplotlib.pyplot as plt
    others = scores.drop(columns=CONTROL)
    gap = (scores[CONTROL] - others.max(axis=1)) * 100
    train_rows = rows.reindex(scores.index) * 0.9
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.axvline(900, color=GRID, linewidth=1.5)
    colours = np.where(gap > TIE * 100, BLUE, np.where(gap < -TIE * 100, RED, GRAY))
    ax.scatter(train_rows, gap, s=36, c=colours, edgecolors=SURFACE, linewidths=1, zorder=3)
    ax.set_xlabel('training rows per fold', color=MUTED, fontsize=9)
    ax.set_ylabel('PDL minus the best other model (pp)', color=MUTED, fontsize=9)
    ax.set_title('Where PDL stands against the best of the 10 other defaults, per task', color=INK, fontsize=11, loc='left')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--figures', default='figures')
    args = parser.parse_args()
    from autorank import autorank, plot_stats

    frame = load(args.results)
    models = [m for m in MODELS if m in set(frame.model)]
    acc, imputed = score_matrix(frame, models)
    fit = frame[frame.status == 'ok'].pivot(index='task', columns='model', values='fit_seconds').reindex(index=CC18_SMALL, columns=models)
    names = frame.dropna(subset=['dataset']).groupby('task')['dataset'].first().reindex(acc.index)
    rows = frame.dropna(subset=['dataset']).groupby('task')['rows'].first()

    print('# Small-data benchmark: statistics\n')
    print(f'{len(acc)} tasks x {len(models)} models. Results imputed with the majority-class score: '
          f'{int(imputed.to_numpy().sum())} of {imputed.size} '
          f'{ {LABELS.get(m, m): int(imputed[m].sum()) for m in models if imputed[m].any()} }\n')

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):  # autorank prints its own summary
        result = autorank(acc.rename(columns=LABELS), alpha=ALPHA, order='descending', force_mode='nonparametric', verbose=False)
    ranks = rank_summary(acc)
    best = ranks.index[0]
    top_group = [m for m in ranks.index if ranks[m] - ranks[best] <= result.cd]
    print(f'Friedman p = {result.pvalue:.2g}; Nemenyi critical difference = {result.cd:.2f} ranks; '
          f'top group (within the CD of {LABELS.get(best, best)}): {[LABELS.get(m, m) for m in top_group]}\n')
    summary = pd.DataFrame({'mean rank': ranks, 'mean accuracy': acc.mean()[ranks.index],
                            'top-3 finishes': (acc.rank(axis=1, ascending=False, method='average') <= 3).sum()[ranks.index],
                            'median fit s': fit.median()[ranks.index],
                            'median predict s': frame[frame.status == 'ok'].groupby('model')['predict_seconds'].median().reindex(ranks.index)})
    print(summary.rename(index=LABELS).to_markdown(floatfmt='.3f'), '\n')

    comparisons = pdl_comparisons(acc, fit)
    shown = comparisons.assign(model=comparisons.model.map(lambda m: LABELS.get(m, m)))
    print('## PDL against each model\n')
    print(shown[['model', 'wins', 'ties', 'losses', 'mean_diff', 'ci_low', 'ci_high', 'wilcoxon_p', 'holm_p', 'p_pdl', 'p_rope',
                 'p_other', 'fit_ratio', 'verdict']].to_markdown(index=False, floatfmt='.3f'), '\n')
    label, place = outcome(ranks, comparisons, top_group)
    print(f'**Declared outcome: {label}** (PDL mean-rank place {place} of {len(models)}).\n')

    print('## Where PDL stands by size\n')
    gap = acc[CONTROL] - acc.drop(columns=CONTROL).max(axis=1)
    for title, mask in [('up to 1,000 rows', rows.reindex(acc.index) <= 1000), ('1,001 to 2,000 rows', rows.reindex(acc.index) > 1000)]:
        r = rank_summary(acc[mask.to_numpy()])
        print(f'- {title} ({int(mask.sum())} tasks): PDL mean rank {r[CONTROL]:.2f} (place {int(r.rank()[CONTROL])}), '
              f'mean gap to the best other model {gap[mask.to_numpy()].mean() * 100:+.1f} pp')
    print()

    print('## Sensitivity\n')
    variants = [(f'Without the {len(DEFAULTS_STUDY)} tasks of the defaults study', acc[~names.isin(DEFAULTS_STUDY).to_numpy()]),
                (f'Also without {", ".join(sorted(OTHER_TUNING))}', acc[~names.isin(DEFAULTS_STUDY | OTHER_TUNING).to_numpy()])]
    mfeat = names.str.startswith('mfeat').to_numpy()
    variants.append(('mfeat views averaged into one task', pd.concat([acc[~mfeat], acc[mfeat].mean().to_frame('mfeat').T])))
    for metric in ('roc_auc', 'macro_f1'):
        variants.append((f'{metric} instead of accuracy', score_matrix(frame, models, metric)[0]))

    def kind(text):  # the verdict without its numbers
        return text.split(' (95%')[0]

    full = comparisons.set_index('model')['verdict'].map(kind)
    for title, data in variants:
        r = rank_summary(data)
        table = pdl_comparisons(data).set_index('model')
        et = table.loc['extra_trees']
        changed = [f'{LABELS.get(m, m)}: {full[m]} -> {kind(table.loc[m, "verdict"])}'
                   for m in table.index if kind(table.loc[m, 'verdict']) != full[m]]
        print(f'- {title} ({len(data)} tasks): PDL mean rank {r[CONTROL]:.2f} (place {int(r.rank()[CONTROL])}); '
              f'against ExtraTrees: {et.wins}/{et.ties}/{et.losses}, Holm p {et.holm_p:.3f}, {et.verdict}. '
              f'Verdicts that change: {"; ".join(changed) if changed else "none"}')
    for tie in (0., 0.01):
        counts = {LABELS.get(m, m): '/'.join(map(str, wtl((acc[CONTROL] - acc[m]).round(4), tie))) for m in models if m != CONTROL}
        print(f'- Wins/ties/losses with a tie threshold of {tie * 100:.0f} pp: {counts}')

    os.makedirs(args.figures, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_stats(result, allow_insignificant=True)
    plt.savefig(os.path.join(args.figures, 'f1_cd_diagram.png'), dpi=200, bbox_inches='tight')
    plt.close('all')
    figure_differences(acc, comparisons, os.path.join(args.figures, 'f2_pdl_minus_each_model.png'))
    figure_rank_vs_time(ranks, fit, os.path.join(args.figures, 'f3_rank_vs_fit_time.png'))
    figure_gap_vs_rows(acc, rows, os.path.join(args.figures, 'f4_gap_vs_rows.png'))
    with open(os.path.join(args.figures, 'comparisons.json'), 'w') as f:
        json.dump(comparisons.to_dict(orient='records'), f, indent=1)


if __name__ == '__main__':
    main()
