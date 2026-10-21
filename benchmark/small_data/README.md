# Small-data benchmark: PDL against popular classifiers

PDL (pdll 1.0, default settings) against ten popular classifiers, each with its default settings, on every
OpenML-CC18 task with at most 2,000 rows: 35 tasks, the official 10-fold split of each task, raw DataFrames.

## Protocol (fixed before any result was looked at)

- **Tasks**: the 35 OpenML-CC18 tasks with at most 2,000 instances (`run_all.py`, `CC18_SMALL`). No task is added or
  dropped after seeing results; results are also reported separately for tasks up to 1,000 rows and above.
- **Models** (`models.py`): PDL, logistic regression, k-NN, SVM (RBF), decision tree, random forest, extra trees,
  HistGradientBoosting, LightGBM, XGBoost, CatBoost. Defaults only, apart from `random_state=0`, thread counts,
  silenced logs, `max_iter=1000` for logistic regression and `probability=True` for the SVM (for ROC AUC).
- **Input**: each model gets the raw DataFrame the way its documentation recommends: native categoricals for
  HistGradientBoosting, LightGBM and XGBoost, `cat_features` for CatBoost, ordinal codes with NaN passed through for
  the scikit-learn trees, one-hot + imputation + scaling for the linear, k-NN and SVM models, and the raw DataFrame
  for PDL.
- **Metrics**: accuracy (OpenML's leaderboard metric), macro-F1 and ROC AUC (one-vs-rest macro for multiclass), the
  mean over the 10 folds; fit and predict seconds.
- **Ranking against OpenML**: `public_rank.py` downloads every public run on each task (same split, predictive
  accuracy) and counts one entry per method (flow name without its version suffix, best run of that method).

## Run it

```shell
pip install -e . -r benchmark/small_data/requirements.txt   # scikit-learn >= 1.6 is required
cd benchmark/small_data
python run_all.py --out results/cc18_small.jsonl --workers 4 --n-jobs 2   # a few hours on a laptop; add --retry-failed to rerun errors and timeouts
python public_rank.py results/cc18_small.jsonl                            # writes results/openml_public.json
python report.py results/cc18_small.jsonl results/openml_public.json > RESULTS.md
```

`run_all.py` skips (task, model) pairs that are already in the output file, so an interrupted run resumes where it
stopped, and it refuses to start when a requested model's package or scikit-learn >= 1.6 is missing.
