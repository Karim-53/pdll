"""The compared classifiers: default hyperparameters, each fed a raw DataFrame the way its documentation recommends.

- PDL gets the raw DataFrame (it ordinal-encodes categories itself).
- HistGradientBoosting, LightGBM and XGBoost use their native categorical support (pandas ``category`` dtype).
- CatBoost gets its categorical columns as strings through ``cat_features``.
- Random forest, extra trees and the decision tree get ordinal-encoded categories; NaN is passed through (scikit-learn
  trees support missing values).
- Logistic regression, k-NN and the SVM get one-hot categories, median-imputed and standardized numbers.

Only deviations from the defaults: ``random_state=0`` where available, the thread count, silenced training logs,
``max_iter=1000`` for logistic regression (the default 100 often stops before convergence) and ``probability=True``
for the SVM (for ROC AUC; predictions are unchanged).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 0
MAX_NATIVE_CATEGORIES = 255  # HistGradientBoosting's limit; wider categorical columns are ordinal-encoded for every model


def categorical_columns(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def _as_category(X, categories):
    """Categorical columns as pandas ``category`` with the training fold's categories (unseen values become NaN);
    columns wider than ``MAX_NATIVE_CATEGORIES`` become their integer codes."""
    X = X.copy()
    for c, known in categories.items():
        values = pd.Categorical(X[c].astype(object), categories=known)
        X[c] = pd.Series(values.codes, index=X.index).where(values.codes >= 0, np.nan) if len(known) > MAX_NATIVE_CATEGORIES else values
    return X


def _as_string(X, columns):
    X = X.copy()
    for c in columns:
        X[c] = X[c].astype(object).where(X[c].notna(), 'nan').astype(str)
    return X


def _ordinal(X):
    cats = categorical_columns(X)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=np.nan)
    return ColumnTransformer([('categorical', encoder, cats)], remainder='passthrough')


def _one_hot_scaled(X):
    cats = categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    to_object = FunctionTransformer(lambda part: part.astype(object).where(part.notna(), np.nan))
    return ColumnTransformer([
        ('categorical', make_pipeline(to_object, SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), cats),
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), nums),
    ])


def _native(model, X):
    categories = {c: pd.Index(X[c].dropna().astype(object).unique()).sort_values() for c in categorical_columns(X)}
    return make_pipeline(FunctionTransformer(lambda part: _as_category(part, categories)), model)


def make_model(name, X, n_jobs):
    """A fresh, unfitted model called ``name`` for the training DataFrame ``X``."""
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if name == 'pdl':
        from pdll import PairwiseDifferenceClassifier
        # identical to PairwiseDifferenceClassifier(random_state=0) apart from the thread count
        return PairwiseDifferenceClassifier(ExtraTreesClassifier(class_weight='balanced', n_jobs=n_jobs), random_state=SEED)
    if name == 'logistic_regression':
        return make_pipeline(_one_hot_scaled(X), LogisticRegression(max_iter=1000))
    if name == 'knn':
        return make_pipeline(_one_hot_scaled(X), KNeighborsClassifier(n_jobs=n_jobs))
    if name == 'svm':
        return make_pipeline(_one_hot_scaled(X), SVC(probability=True, random_state=SEED))
    if name == 'decision_tree':
        return make_pipeline(_ordinal(X), DecisionTreeClassifier(random_state=SEED))
    if name == 'random_forest':
        return make_pipeline(_ordinal(X), RandomForestClassifier(random_state=SEED, n_jobs=n_jobs))
    if name == 'extra_trees':
        return make_pipeline(_ordinal(X), ExtraTreesClassifier(random_state=SEED, n_jobs=n_jobs))
    if name == 'hist_gradient_boosting':  # 'from_dtype' is the default from scikit-learn 1.6 on; spelled out for clarity
        return _native(HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=SEED), X)
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        return _native(LGBMClassifier(random_state=SEED, n_jobs=n_jobs, verbose=-1), X)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return _native(XGBClassifier(enable_categorical=True, random_state=SEED, n_jobs=n_jobs), X)
    if name == 'catboost':
        from catboost import CatBoostClassifier
        cats = categorical_columns(X)
        model = CatBoostClassifier(random_seed=SEED, thread_count=n_jobs, verbose=0, allow_writing_files=False, cat_features=cats)
        return make_pipeline(FunctionTransformer(lambda part: _as_string(part, cats)), model)
    raise ValueError(f'unknown model {name!r}')


MODELS = ['pdl', 'logistic_regression', 'knn', 'svm', 'decision_tree', 'random_forest', 'extra_trees',
          'hist_gradient_boosting', 'lightgbm', 'xgboost', 'catboost']
EXTRA_PACKAGES = {'lightgbm': 'lightgbm', 'xgboost': 'xgboost', 'catboost': 'catboost', 'tabpfn': 'tabpfn'}  # model -> module
MIN_SKLEARN = (1, 6)


def check_environment(models):
    """Fail early, with the install command, instead of recording every run of a missing package as a model error."""
    import importlib.util

    import sklearn
    problems = []
    if tuple(int(part) for part in sklearn.__version__.split('.')[:2]) < MIN_SKLEARN:
        problems.append(f'scikit-learn {sklearn.__version__} is too old: this benchmark needs >= 1.6 (native categoricals in '
                        f'HistGradientBoosting, missing values in every tree)')
    missing = [EXTRA_PACKAGES[m] for m in models if m in EXTRA_PACKAGES and importlib.util.find_spec(EXTRA_PACKAGES[m]) is None]
    if missing:
        problems.append(f'missing packages for the requested models: {", ".join(missing)}')
    if problems:
        raise SystemExit('; '.join(problems) + '. Install the benchmark environment with: '
                         'pip install -e . -r benchmark/small_data/requirements.txt')
