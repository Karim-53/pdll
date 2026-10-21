# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-21

A rewrite for robustness and speed. On the inputs the paper used (numpy arrays), predictions are identical to
0.3.1 with the same `use_prior`; `tests/test_golden.py` checks this.

### Changed (breaking)
- `use_prior="auto"` takes the prior-based posterior only when a class has fewer than 5 training rows **and** the
  classes are about equally frequent (two classes, or the largest at most twice the smallest); class sizes count
  `sample_weight` as repeated rows (so `BaggingClassifier` members decide like the full model), `class_weight` does not
  change them. 0.3.1 took it whenever a class had fewer than 5 rows. On imbalanced multiclass data that posterior
  favours large classes so strongly that a tiny class was never predicted, not even on its own training rows (with
  40/40/40/10/3 rows: accuracy 0.58 against 0.85; on 50 long-tailed classes: 0.06 against 0.76). `use_prior=True`
  still gives the paper's posterior.
- `predict` returns a numpy array of labels and `predict_proba` a numpy array whose columns follow `classes_`,
  like every scikit-learn classifier. 0.3.1 returned a pandas Series/DataFrame for DataFrame input, so
  `predict_proba(X)[:, 1]` failed.
- The base estimator is cloned into `estimator_`; the object you pass is never refit.
- `use_prior`, `max_anchors`, `max_pairs` and `random_state` are constructor parameters, so `get_params`,
  `set_params`, `clone` and `GridSearchCV` see them. `fit(check_input=...)` is gone.
- Fitted attributes follow scikit-learn: `estimator_`, `classes_`, `n_classes_` (was `nb_classes_`), `prior_`
  (was `prior`), `n_features_in_`, `feature_names_in_`, `X_anchors_`, `y_anchors_`. `X_train_`/`y_train_` are gone.
- Invalid input raises `ValueError`/`TypeError` with a message, never `AssertionError`. Warnings use `warnings.warn`.
- Requires Python ≥ 3.9, numpy ≥ 1.23, pandas ≥ 1.5, scikit-learn ≥ 1.2, scipy ≥ 1.9. Packaging moved to `pyproject.toml`.

### Added
- DataFrames with string, object, category, bool, uint and nullable columns: categorical columns are ordinal-encoded
  inside the estimator (missing/unseen values become -1), and identifier-like columns trigger a warning.
- polars and pyarrow tables (and targets) are accepted and handled like pandas DataFrames, also by
  `PDCDataTransformer`. A polars LazyFrame is collected; without pyarrow (which polars' `to_pandas` needs), polars
  data is converted column by column. Other objects with a `to_pandas` method are left to scikit-learn as before.
- Actionable errors for date, time and timedelta values stored as objects (e.g. pyarrow `date32` columns) instead of
  silently encoding them as categories, for duplicate column names, for cells holding lists or dicts, and for missing
  values in string, object or nullable targets.
- NaN support whenever the base estimator supports it (e.g. HistGradientBoosting).
- `max_pairs` (default `"auto"`: 100,000 pairs, fewer for very wide data) and `max_anchors` (default 500) bound training
  and prediction cost. Up to 316 training rows the behaviour is exactly the paper's; `max_pairs=None, max_anchors=None`
  reproduces it at any size, which `run_benchmark.py` now does. The defaults come from a study on 20 real datasets: against
  1M pairs, mean macro-F1 -0.005 for about 10x less time and memory, and no out-of-memory failure on a 16 GB machine
  (1M pairs ran out of memory on a 1,473-row dataset and needed 5 GB for Titanic).
- `sample_weight` in `fit` for both estimators (pair weight w_i·w_j, equivalent to repeating rows).
- `class_weight` (`None`, `"balanced"` or a dict) for the classifier, applied as sample weights. On imbalanced data
  PDL otherwise under-predicts minority classes. With weights, a base estimator with `class_weight="balanced"` balances
  "same" against "different" pairs on the weighted counts, so every scikit-learn version fits the same model (before
  1.8, forests balanced on row counts and ignored the weights).
- `fit` warns above 2,000 training rows: PDL is a small-data method, and above that size the base estimator alone was
  usually more accurate in the real-data benchmark.
- `predict_log_proba`, `pdll.__citation__`, `CITATION.cff` (GitHub "Cite this repository").
- `benchmark/real_world`: validation on 131 real datasets (OpenML-CC18, OpenML-CTR23, Kaggle, scikit-learn, Zindi):
  129 ran, 2 were refused with an actionable message, none failed.
- Tests: golden equivalence with 0.3.1, scikit-learn's `parametrize_with_checks`, a reference implementation of the
  paper's formulas, an end-to-end accuracy/speed regression gate for the classifier and the regressor, score floors
  (95% of the measured score, 90% for R²) on every unit test that computes a score, and OpenML integration tests.
  CI covers the oldest supported, Kaggle-like, Colab-like and latest dependency stacks on Linux, Windows and macOS.

### Fixed
- Wrong predictions when `X`/`y` are pandas objects whose index is not 0..N-1 (e.g. straight out of
  `train_test_split`); a string index crashed.
- String labels, or labels that are not 0..K-1, crashed when a class had fewer than 5 rows (prior path).
- Fitting on numpy and predicting on a DataFrame (or the reverse) raised `KeyError`.
- The class prior was not recomputed when refitting.
- bool, uint and category features were rejected; uint differences could wrap around.
- `PDCDataTransformer` imported `benchmark`, which is not part of the package, and modified its parameters in `fit`.
- `PairwiseDifferenceRegressor.predict` overwrote zero anchor weights in `sample_weight_` with NaN.
- `PairwiseDifferenceRegressor.fit` silently ignored `sample_weight`.
- Unseen categories at prediction time used a pandas construction that pandas 3 deprecates and pandas 4 will reject.
- Multi-output targets and targets with missing values now get errors that say what to do
  (`MultiOutputClassifier`/`MultiOutputRegressor`, drop rows with a missing target).
- When pairs are subsampled (more than 316 training rows by default), a class could end up without any same-class
  pair, e.g. a single-row class, and was then never predicted, even on its own row; with `class_weight="balanced"`
  its anchor even voted against it. Such a class now keeps the diagonal pairs (i, i) of up to 10 of its rows, so the
  extra pairs are bounded by the number of classes. Datasets where every class has
  same-class pairs, which is nearly all, get exactly the same model as before.
- Numbers stored as text (`read_csv(dtype=str)`, padded `" 5 "`) were encoded as string-ordered categories, so every
  unseen value lost its meaning; they are now converted to numbers, as scikit-learn does. Mostly-numeric text with
  placeholders such as `"?"` or `" "` stays categorical with a warning that says how to convert it.
- Complex numbers lost their imaginary part silently; they are now refused, as by scikit-learn.
- An extra, renamed or missing column at predict time (e.g. Kaggle's `Id`) gave a misleading "pass a DataFrame"
  error; it now gets scikit-learn's feature-name message. A text target for the regressor is told to use the
  classifier.
- `PairwiseDifferenceRegressor`: float16 targets overflowed in the pair differences; `learn_anchor_weights` with a
  column-vector `y_val` built an n_val × n_val × n_anchors array and crashed.
- Prediction is vectorised and memory-bounded: about 2x faster than 0.3.1 at a few hundred anchors. Training cost is
  dominated by the base estimator (unchanged), and `max_pairs` now caps it on larger datasets.

### Removed
- `PairwiseDifferenceBase` and its static helpers (`pair_input`, `pair_output`, ...).

## [0.3.1]
### Fixed
- Fix the bug in the PDC.predict() output

## [0.2.0] - 2024-07-15
### Added
- Add weighting method and their benchmark
- PDC.predict_proba: use norm rather than softmax

## [0.1.0] - 2024-01-01
### Added
- Pairwise Difference Regressor (PDR)
- Initial release of the project.

[1.0.0]: https://github.com/Karim-53/pdll/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Karim-53/pdll/releases/tag/v0.1.0
