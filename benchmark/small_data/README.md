# Small-data benchmark: PDL against popular classifiers

**Results: [RESULTS.md](RESULTS.md).**

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
- **Ranking against OpenML**: `public_rank.py` downloads the public runs on each task (same split, predictive
  accuracy), best first, down to 0.01 below PDL's accuracy, which is all the ranking needs (at most 200,000 runs per
  task; a task that hits this cap shows its rank as ">= n"). It counts one entry per method (flow name without its
  version suffix, best run of that method).

## Run it

```shell
pip install -e . -r benchmark/small_data/requirements.txt autorank==1.3.0 baycomp==1.0.3   # scikit-learn >= 1.6 is required
cd benchmark/small_data
python run_all.py --out results/cc18_small.jsonl --workers 1 --n-jobs 12 --timeout 5400   # the published setting; several hours
python public_rank.py results/cc18_small.jsonl                                            # needs openml.org
python report.py results/cc18_small.jsonl results/openml_public.json > report.md          # ranks and the per-task table
python analyze.py results/cc18_small.jsonl --figures figures > ANALYSIS.md               # statistics and figures
```

`RESULTS.md` is the written summary of `report.md` and `ANALYSIS.md`. `run_all.py` skips (task, model) pairs that
are already in the output file, so an interrupted run resumes where it stopped (`--retry-failed` also reruns errors
and timeouts), and it refuses to start when a requested model's package or scikit-learn >= 1.6 is missing.
