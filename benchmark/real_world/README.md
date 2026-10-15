# pdll on 131 real datasets

Before releasing pdll 1.0, we ran it on 131 real datasets, raw, the way a user would: strings, categories, missing
values and odd dtypes included. The full table is in [RESULTS.md](RESULTS.md) and the raw results are in
[`results/`](results/).

| Source | Datasets |
|---|---|
| [OpenML-CC18](https://www.openml.org/s/99) (curated classification suite) | 72 |
| [OpenML-CTR23](https://www.openml.org/s/353) (curated regression suite) | 35 |
| Kaggle datasets, through their OpenML mirrors (Titanic, House Prices, Telco churn, credit-card fraud, ...) | 13 |
| scikit-learn's bundled and fetched datasets | 10 |
| Zindi (AutoInland vehicle insurance, a private copy that is not redistributed) | 1 |

## Protocol

- OpenML datasets use fold 0 of their **official task split**, the split behind OpenML's public leaderboards.
  The other datasets use a stratified 75/25 split. Test sets are capped at 2,000 rows.
- pdll runs with its **default settings** on the raw DataFrame: `PairwiseDifferenceClassifier(random_state=0)` or
  `PairwiseDifferenceRegressor(random_state=0)`.
- The reference is pdll's own base learner alone: `ExtraTreesClassifier(class_weight="balanced")` or
  `ExtraTreesRegressor()`, on the same data with non-numeric columns ordinal-encoded.
- Each dataset runs in its own process, with a 30-minute timeout and a memory watchdog (11 GB), on a 16 GB laptop.
- We drop identifier columns (Titanic's names and tickets, the `Id` of House Prices, ...) and rows with a missing
  target, as a user would.

## Findings

- **Robustness.** 129 of 131 datasets ran. Two were refused with an actionable error: sparse text
  (20newsgroups) and a 3-target regression (linnerud, which points to `MultiOutputRegressor`). Labels stored as bytes
  (KDD Cup 99) raise scikit-learn's own error, which asks to convert them. There were no crashes, out-of-memory errors
  or timeouts, from 150 rows to 436,000 rows and up to 3,072 features. The median fit took 7 s (classification) and
  29 s (regression), and peak memory stayed at or below 2.6 GB.
- **Accuracy depends on size.** Macro-F1 for classification and R² for regression, PDL minus base learner:

  | Training rows | Classification: mean gap (wins / ties / losses) | Regression: mean gap (wins / ties / losses) |
  |---|---|---|
  | up to 1,000 | -0.001 (10 / 10 / 10) | +0.001 (3 / 4 / 2) |
  | 1,000 to 2,000 | -0.012 (3 / 5 / 10) | +0.002 (4 / 1 / 2) |
  | more than 2,000 | -0.061 (5 / 3 / 35) | -0.033 (5 / 4 / 13) |

  PDL is a small-data method: it ties a strong tree ensemble on average up to about 1,000 rows and wins on many
  datasets. Above 2,000 rows it sees only a sample of the pairs and anchors, and the base learner alone is usually
  better, so `fit` warns from 2,000 rows on. Weak learners gain most, e.g. a single decision tree on 300 digits:
  0.70-0.75 → 0.86-0.91 accuracy (see `notebooks/colab_quickstart.ipynb`).
- **Imbalanced classes.** PDL's ROC AUC is on par with or better than the base learner's, but by default it
  under-predicts minority classes. `class_weight="balanced"` raised macro-F1 by up to 0.05 on 10 imbalanced datasets,
  and never lowered it (`imbalance_probe.py`).

## What the campaign changed in pdll 1.0

- **Defaults** (`defaults_study.sh`, `compare_configs.py`). The original 1M pairs and 1,000 anchors needed 5.4 GB
  for Titanic, ran out of memory on a 1,473-row dataset and timed out on House Prices. The new defaults,
  `max_pairs="auto"` (100,000 pairs) and `max_anchors=500`, cost 0.005 mean macro-F1 for about 10x less time and
  memory. Larger tree leaves were also tried and lost accuracy.
- **Bugs fixed.** Unseen categories used a pandas construction that pandas 4 will reject. Multi-output and
  missing-target errors are now actionable. `class_weight` was added, and so was the warning above 2,000 rows.

## Reproduce

```shell
pip install -e ".[test]" openml psutil tabulate
python benchmark/real_world/run_all.py --out results/campaign.jsonl --split openml-task   # a few hours
python benchmark/real_world/report.py results/campaign.jsonl > RESULTS.md
python benchmark/real_world/openml_leaderboard.py --out results/leaderboard.jsonl 31 37 ...   # 10-fold, vs public runs
```
