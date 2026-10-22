"""TabMini results as Markdown: mean rank of per-dataset ROC AUC, statistics, robustness views and the other metrics.

    python analyze.py results/tabmini.jsonl [--svm-check] > RESULTS.md

Ranks: per dataset, 1 = best mean ROC AUC over the 3 folds, ties averaged; a failed run ranks last (there were none).
``--svm-check`` refits the SVM (a few seconds) to re-score it on ``decision_function``, which scikit-learn's
``roc_auc`` scorer, and so TabMini's ``cross_val_score``, uses for an SVC.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from run_all import DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = 'pdl'
LABELS = {'pdl': 'PDL (pdll 1.0.0)', 'catboost': 'CatBoost', 'random_forest': 'Random forest', 'extra_trees': 'Extra trees',
          'svm': 'SVM (RBF)', 'logistic_regression': 'Logistic regression', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
          'hist_gradient_boosting': 'HistGradientBoosting', 'knn': 'k-NN', 'decision_tree': 'Decision tree'}
VIEWS = {
    # the second dataset of each duplicate pair (same UCI problem, rows reordered)
    '40 distinct datasets': [n for n in DATASETS if n not in {'house_votes_84', 'hungarian', 'heart_c', 'horse_colic'}],
    # TabMini's README, 2025 curation
    "the 25 datasets TabMini now keeps": [
        'analcatdata_aids', 'analcatdata_asbestos', 'analcatdata_bankruptcy', 'analcatdata_creditscore', 'analcatdata_fraud',
        'analcatdata_japansolvent', 'labor', 'lupus', 'parity5', 'postoperative_patient_data', 'appendicitis', 'backache',
        'corral', 'glass2', 'hepatitis', 'molecular_biology_promoters', 'mux6', 'prnn_crabs', 'biomed', 'breast_cancer',
        'prnn_synth', 'bupa', 'haberman', 'irish', 'saheart'],
    # without every dataset whose rows PDL had been run on before (the PDL paper's datasets, pdll's real-data campaign)
    '30 datasets PDL had never been run on': [
        'parity5', 'analcatdata_fraud', 'analcatdata_aids', 'analcatdata_bankruptcy', 'analcatdata_japansolvent', 'labor',
        'analcatdata_asbestos', 'lupus', 'postoperative_patient_data', 'analcatdata_cyyoung9302', 'analcatdata_cyyoung8092',
        'analcatdata_creditscore', 'appendicitis', 'analcatdata_boxing1', 'hepatitis', 'corral', 'backache', 'prnn_crabs',
        'sonar', 'biomed', 'analcatdata_lawsuit', 'breast_cancer', 'bupa', 'colic', 'horse_colic', 'house_votes_84', 'vote',
        'saheart', 'clean1', 'irish'],
    # not in TabMini's list of datasets used for TabPFN's prior (tabmini/data/data_info.py)
    "23 datasets outside TabPFN's prior": [
        'parity5', 'analcatdata_fraud', 'analcatdata_aids', 'analcatdata_bankruptcy', 'analcatdata_japansolvent', 'labor',
        'analcatdata_asbestos', 'lupus', 'postoperative_patient_data', 'analcatdata_cyyoung9302', 'analcatdata_cyyoung8092',
        'analcatdata_creditscore', 'appendicitis', 'molecular_biology_promoters', 'analcatdata_boxing1', 'mux6',
        'analcatdata_boxing2', 'glass2', 'hungarian', 'bupa', 'horse_colic', 'house_votes_84', 'clean1'],
}


def read(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return pd.DataFrame(rows).drop_duplicates(['dataset', 'model'], keep='last')


def table_of(df, metric, higher_is_better=True):
    ok = df[df.status == 'ok'].pivot(index='name', columns='model', values=metric).reindex(index=DATASETS, columns=list(LABELS))
    return ok if higher_is_better else -ok


def ranks_of(table):
    filled = table.fillna(-np.inf)
    return filled.apply(lambda row: pd.Series(rankdata(-row.to_numpy(), method='average'), index=row.index), axis=1)


def holm(pvalues):
    order, adjusted, running = np.argsort(pvalues), np.empty(len(pvalues)), 0.
    for k, i in enumerate(order):
        running = max(running, min(1., (len(pvalues) - k) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def place(table):
    means = ranks_of(table).mean().sort_values()
    return list(means.index).index(ENTRY) + 1, means


def main_table(table):
    ranks = ranks_of(table)
    means = ranks.mean().sort_values()
    print('| Place | Model | Mean rank | Mean AUC | Top-3 finishes |\n|---|---|---|---|---|')
    for i, (model, r) in enumerate(means.items(), 1):
        bold = '**' if model == ENTRY else ''
        print(f'| {i} | {bold}{LABELS[model]}{bold} | {bold}{r:.2f}{bold} | {table[model].mean():.3f} | {int((ranks[model] <= 3).sum())} |')
    stat, p = friedmanchisquare(*[table[m].to_numpy() for m in table.columns])
    print(f'\nFriedman test: chi² = {stat:.1f}, p = {p:.1e}.')
    try:
        from autorank import autorank
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            result = autorank(table, alpha=0.05, verbose=False, force_mode='nonparametric')
        best = result.rankdf['meanrank'].min()
        group = [LABELS[m] for m in result.rankdf.index if result.rankdf.loc[m, 'meanrank'] - best <= result.cd]
        print(f'Nemenyi critical difference: {result.cd:.2f} ranks; within it of the best: {", ".join(reversed(group))}.')
    except ImportError:
        print('(install autorank for the Nemenyi critical difference)')
    others = [m for m in means.index if m != ENTRY]
    raw = [wilcoxon(table[ENTRY] - table[m], zero_method='zsplit').pvalue for m in others]
    print(f'\n| PDL vs | Wins / ties / losses (ties within 0.005 AUC) | Mean AUC difference | Wilcoxon p, Holm-corrected over 10 |\n|---|---|---|---|')
    for m, p_adj in zip(others, holm(np.array(raw))):
        d = table[ENTRY] - table[m]
        print(f'| {LABELS[m]} | {(d > 0.005).sum()} / {(d.abs() <= 0.005).sum()} / {(d < -0.005).sum()} | {d.mean():+.4f} | {p_adj:.3g} |')


def svm_decision_function_auc():
    from sklearn.metrics import roc_auc_score
    from models import make_model
    from run_one import folds, load
    scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name in DATASETS:
            X, y = load(name)
            aucs = []
            for train, test in folds(y):
                model = make_model('svm', X.iloc[train], 1).fit(X.iloc[train], y[train])
                aucs.append(roc_auc_score(y[test], model.decision_function(X.iloc[test])))
            scores[name] = np.mean(aucs)
    return pd.Series(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--svm-check', action='store_true')
    args = parser.parse_args()
    df = read(args.results)
    auc = table_of(df, 'roc_auc')
    failed = int(auc.isna().sum().sum())
    print('# PDL on TabMini: results\n')
    print(f'{len(DATASETS)} binary datasets with 32–500 rows, 11 classifiers at default settings, TabMini\'s protocol (one shuffle '
          f'with seed 42, stratified 3-fold CV, ROC AUC per fold, averaged). Failed runs: {failed}. Generated by `analyze.py` '
          f'from [`results/tabmini.jsonl`](results/tabmini.jsonl).\n')
    print('## All 44 datasets: mean rank of per-dataset ROC AUC\n')
    main_table(auc)
    print('\n## Robustness\n')
    print('| View | Datasets | PDL\'s place | PDL\'s mean rank | 1st (mean rank) |\n|---|---|---|---|---|')
    for title, names in VIEWS.items():
        p, means = place(auc.loc[names])
        print(f'| {title} | {len(names)} | {p} | {means[ENTRY]:.2f} | {LABELS[means.index[0]]} ({means.iloc[0]:.2f}) |')
    p, means = place(auc.round(2))
    print(f'| AUC rounded to 2 decimals (TabMini\'s convention) | 44 | {p} | {means[ENTRY]:.2f} | {LABELS[means.index[0]]} ({means.iloc[0]:.2f}) |')
    worst = len(DATASETS)
    for k in (1, 2):
        from itertools import combinations
        worst = max(place(auc.drop(index=list(c)))[0] for c in combinations(DATASETS, k))
        print(f'| Leave any {k} dataset{"s" if k > 1 else ""} out (worst case) | {44 - k} | {worst} | | |')
    rng = np.random.default_rng(0)
    ranks = ranks_of(auc).to_numpy()
    entry = list(LABELS).index(ENTRY)
    places = []
    for _ in range(10_000):
        means = ranks[rng.integers(0, len(DATASETS), len(DATASETS))].mean(axis=0)
        places.append(1 + int((means < means[entry]).sum()))
    places = np.array(places)
    print(f'\nBootstrap over datasets (10,000 resamples, seed 0): PDL 1st in {np.mean(places == 1):.0%}, 2nd in '
          f'{np.mean(places == 2):.0%}, top 3 in {np.mean(places <= 3):.0%}.')
    if args.svm_check:
        checked = auc.copy()
        checked['svm'] = svm_decision_function_auc()
        p, means = place(checked)
        print(f'\nSVM re-scored on `decision_function` (scikit-learn\'s `roc_auc` scorer): PDL {p} (mean rank '
              f'{means[ENTRY]:.2f}), SVM {list(means.index).index("svm") + 1} ({means["svm"]:.2f}), SVM mean AUC '
              f'{checked["svm"].mean():.3f}.')
    print('\n## Other metrics (recorded, no claim made on them)\n')
    print('| Metric | PDL\'s place | PDL\'s mean rank | 1st (mean rank) |\n|---|---|---|---|')
    for metric, label, higher in (('accuracy', 'Accuracy', True), ('macro_f1', 'Macro-F1', True), ('log_loss', 'Log loss', False)):
        p, means = place(table_of(df, metric, higher))
        print(f'| {label} | {p} | {means[ENTRY]:.2f} | {LABELS[means.index[0]]} ({means.iloc[0]:.2f}) |')
    fit = df[df.status == 'ok'].groupby('model').fit_seconds.median()
    print(f'\nMedian training time per dataset (3 folds, 6 threads, a laptop also running other jobs; indicative): PDL '
          f'{fit[ENTRY]:.1f} s, CatBoost {fit["catboost"]:.1f} s, random forest {fit["random_forest"]:.1f} s.')
    print('\n## Per dataset (mean ROC AUC)\n')
    print(auc.rename(columns=LABELS).round(3).to_markdown())


if __name__ == '__main__':
    sys.exit(main())
