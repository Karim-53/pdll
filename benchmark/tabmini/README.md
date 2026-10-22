# pdll on TabMini: 44 small binary datasets

[TabMini](https://github.com/RicardoKnauer/TabMini) (Knauer et al., 2024, [arXiv 2409.01635](https://arxiv.org/abs/2409.01635))
is a benchmark of 44 binary classification datasets from PMLB with 32 to 500 rows. We ran pdll 1.0.0 at its default
settings against ten common classifiers at theirs, with TabMini's own protocol.

## Result

pdll 1.0.0 ranked **2nd of 11 untuned classifiers by mean rank of per-dataset ROC AUC** (3.98; lower is better),
behind CatBoost (3.65) and with a better mean rank than random forest (4.59), extra trees, SVM, logistic regression,
LightGBM, XGBoost, HistGradientBoosting, k-NN and a decision tree.

| Place | Model | Mean rank | Top-3 finishes (of 44) |
|---|---|---|---|
| 1 | CatBoost | 3.65 | 22 |
| **2** | **PDL (pdll 1.0.0)** | **3.98** | **23** |
| 3 | Random forest | 4.59 | 11 |
| 4 | Extra trees | 5.08 | 13 |
| 5 | SVM (RBF) | 5.15 | 15 |

This is a placing, not a win. PDL's differences from CatBoost, random forest, extra trees, SVM, logistic regression and
XGBoost are not statistically significant (Wilcoxon signed-rank, Holm-corrected; CatBoost p = 0.83). It is significantly
better than HistGradientBoosting, k-NN and the decision tree, and than LightGBM by the Wilcoxon test only.

The place is stable. PDL is 2nd on the 40 distinct datasets, on the 25 that TabMini now keeps, on the 30 datasets PDL
had never been run on before, on the 23 outside TabPFN's prior, with AUC rounded to 2 decimals, and when any one or two
datasets are left out. Resampling the datasets, it comes 2nd in 65% of resamples and in the top 3 in 98%. Its median
training time per dataset was 4.6 s, against 11.5 s for CatBoost. Full tables: [RESULTS.md](RESULTS.md).

## Protocol

- **Data:** TabMini's 44 PMLB files, from PMLB commit 872ec0a (9 of them were renamed `_deprecated_*` later), NaN
  rows dropped, rows shuffled with `sample(frac=1, random_state=42)`, positive class = the largest label.
  `check_data.py` confirms the data are identical, row for row, to the snapshot in TabMini's repository.
- **Splits and metric:** `StratifiedKFold(3)`, which is what TabMini's `cv=3` means for a classifier; ROC AUC per test
  fold, averaged over the 3 folds.
- **Models:** the classifiers of the [small-data benchmark](../small_data/models.py), each with its default
  hyperparameters, `random_state=0` and the input handling its documentation recommends. PDL is
  `PairwiseDifferenceClassifier(random_state=0)`; the code installed is identical to the pdll 1.0.0 wheel on PyPI.
  No model was tuned.
- **Ranking:** per dataset, 1 = best mean AUC, ties averaged; mean over datasets. Friedman test, Nemenyi critical
  difference, Wilcoxon signed-rank against each competitor with Holm correction over 10 comparisons.

The models, splits, metric, ranking and statistics were fixed before any model was fitted on TabMini. ROC AUC, binary
targets and small data were chosen after an earlier development benchmark (35 OpenML-CC18 tasks with 500–2,000 rows),
on which PDL did better by ROC AUC than by accuracy or macro-F1, and better on binary and smaller tasks.

## Limits

- Binary classification with at most 500 rows only; one split seed and one model seed; no tuning for any model.
- Every model is scored on `predict_proba`. scikit-learn's `roc_auc` scorer would use the SVM's `decision_function`;
  re-scored that way, PDL is still 2nd (3.99) and the SVM 5th (`analyze.py --svm-check`).
- By mean AUC instead of mean rank, PDL is 3rd (0.843; SVM 0.850, CatBoost 0.850).
- Default LightGBM and HistGradientBoosting cannot split on the 6 smallest datasets (at most 57 rows) and score 0.5
  there, which accounts for part of PDL's lead over them.
- Four pairs of datasets are the same UCI problem (vote / house_votes_84, heart_h / hungarian, cleve / heart_c,
  colic / horse_colic); see the 40-dataset view.
- TabPFN and AutoML tools were not part of this comparison.
- Accuracy, macro-F1 and log loss were recorded too (PDL 3rd on each); no claim is made on them.

## Reproduce

```shell
pip install -e . -r benchmark/small_data/requirements.txt
cd benchmark/tabmini
python check_data.py                                     # data identical to TabMini's snapshot
python run_all.py --out results/tabmini.jsonl            # about an hour on a laptop, CatBoost dominating
python analyze.py results/tabmini.jsonl --svm-check > RESULTS.md
```
