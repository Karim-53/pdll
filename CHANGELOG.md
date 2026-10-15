# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-15

A rewrite for robustness and speed. On the inputs the paper used (numpy arrays), predictions are identical to
0.3.1; `tests/test_golden.py` checks this.

### Changed (breaking)
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
- NaN support whenever the base estimator supports it (e.g. HistGradientBoosting).
- `max_pairs` (default 1,000,000) and `max_anchors` (default 1,000) bound training and prediction cost. Below these
  limits the behaviour is exactly the paper's.
- `sample_weight` in `fit` for both estimators (pair weight w_i·w_j, equivalent to repeating rows).
- `predict_log_proba`, `pdll.__citation__`, `CITATION.cff` (GitHub "Cite this repository").
- Tests: golden equivalence with 0.3.1, scikit-learn's `parametrize_with_checks`, a reference implementation of the
  paper's formulas, an end-to-end accuracy/speed regression gate, and OpenML integration tests. CI covers the oldest
  supported, Kaggle-like, Colab-like and latest dependency stacks on Linux, Windows and macOS.

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
