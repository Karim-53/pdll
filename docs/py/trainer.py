"""In-browser training core for the PDL demo (runs inside Pyodide).

Everything here is plain Python on numpy/pandas/scikit-learn, so it can also
be executed with CPython for testing:  python -m trainer  (see __main__).
"""
import io
import json
import pickle
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SEED = 53                 # reproducible sampling / splits
MAX_TRAIN = 599           # quick-demo hard cap requested for this PoC
PAIR_BUDGET = 6_000_000   # max cells (rows*cols) of the pairwise train matrix
MIN_ROWS = 12
PAPER_URL = 'https://arxiv.org/abs/2406.20031'

_progress_cb = None       # set by the worker: called with a JSON string


def set_progress_callback(cb):
    global _progress_cb
    _progress_cb = cb


def _emit(obj):
    if _progress_cb is not None:
        _progress_cb(json.dumps(obj))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class S:
    train_df = None
    test_df = None
    train_name = ''
    test_name = ''
    prep = None           # dict produced by prepare()
    models = {}           # model_id -> fitted estimator
    results = {}          # model_id -> result dict


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
def _read_any(data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    buf = io.BytesIO(data)
    if name.endswith(('.xlsx', '.xlsm', '.xls')):
        return pd.read_excel(buf)
    if name.endswith('.json'):
        try:
            return pd.read_json(io.BytesIO(data))
        except ValueError:
            return pd.read_json(io.BytesIO(data), lines=True)
    if name.endswith('.parquet'):
        raise ValueError('Parquet is not supported in the browser demo. '
                         'Please export your data as CSV, TSV, XLSX or JSON.')
    # csv / tsv / txt / unknown -> sniff the separator
    for enc in ('utf-8', 'latin-1'):
        try:
            buf.seek(0)
            df = pd.read_csv(io.TextIOWrapper(buf, encoding=enc), sep=None, engine='python')
            if df.shape[1] == 0:
                raise ValueError('no columns detected')
            return df
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f'Could not parse "{filename}". Supported formats: CSV, TSV, TXT, XLSX, JSON.')


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # drop a pandas index column that sneaks in from to_csv(index=True)
    for c in list(df.columns):
        if c.startswith('Unnamed: 0') and df[c].is_unique and pd.api.types.is_integer_dtype(df[c]):
            df = df.drop(columns=[c])
    return df


def load_files(train_bytes, train_name, test_bytes=None, test_name=None) -> str:
    """Load uploaded file(s); returns the analysis JSON."""
    S.prep = None
    S.models = {}
    S.results = {}
    S.train_df = _clean_columns(_read_any(bytes(train_bytes), train_name))
    S.train_name = train_name
    if test_bytes is not None:
        S.test_df = _clean_columns(_read_any(bytes(test_bytes), test_name))
        S.test_name = test_name
    else:
        S.test_df = None
        S.test_name = ''
    return analyze()


def load_demo(which: str) -> str:
    """Built-in example datasets (generated locally by scikit-learn)."""
    from sklearn import datasets
    rng = np.random.RandomState(SEED)
    if which == 'iris':
        d = datasets.load_iris(as_frame=True)
        df = d.frame.rename(columns={'target': 'target'})
        df['target'] = d.target_names[d.target]
        name = 'iris (demo)'
    elif which == 'diabetes':
        d = datasets.load_diabetes(as_frame=True)
        df = d.frame
        name = 'diabetes (demo)'
    elif which == 'blobs':
        X, y = datasets.make_classification(
            n_samples=400, n_features=8, n_informative=5, n_classes=3,
            weights=[0.6, 0.3, 0.1], random_state=SEED)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = np.array(['alpha', 'beta', 'gamma'])[y]
        # sprinkle some missing values to demo the handling
        mask = rng.random(df.shape) < 0.03
        mask[:, -1] = False
        df = df.mask(pd.DataFrame(mask, columns=df.columns))
        name = 'synthetic imbalanced (demo)'
    else:
        raise ValueError(f'unknown demo {which}')
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    S.prep = None
    S.models = {}
    S.results = {}
    S.train_df = df
    S.train_name = name
    S.test_df = None
    S.test_name = ''
    return analyze()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _find_target(df: pd.DataFrame):
    for c in df.columns:
        if c.strip().lower() == 'target':
            return c
    return None


def _detect_task(y: pd.Series) -> str:
    y = y.dropna()
    if pd.api.types.is_bool_dtype(y) or y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype):
        return 'classification'
    n = y.nunique()
    if pd.api.types.is_integer_dtype(y) and n <= 20:
        return 'classification'
    if n <= 5:
        return 'classification'
    return 'regression'


def analyze(target=None, task=None) -> str:
    df = S.train_df
    out = {'ok': True, 'blockers': [], 'warnings': [], 'trainName': S.train_name, 'testName': S.test_name}
    if df is None or len(df) == 0:
        return json.dumps({'ok': False, 'blockers': ['The training file is empty.']})

    tgt = target or _find_target(df)
    out['columns'] = [str(c) for c in df.columns]
    out['targetFound'] = tgt is not None
    out['target'] = tgt
    out['nRows'] = int(len(df))
    out['nCols'] = int(df.shape[1])

    if tgt is None:
        out['warnings'].append(
            "No column named 'target' was found. Pick the target column below "
            "(everything else will be used as input features).")
        return json.dumps(out)

    y = df[tgt]
    detected = _detect_task(y)
    out['taskDetected'] = detected
    out['task'] = task or detected

    n_missing_y = int(y.isna().sum())
    if n_missing_y:
        out['warnings'].append(
            f'{n_missing_y} row(s) have a missing target value and will be dropped.')
    yv = y.dropna()
    if len(df) - n_missing_y < MIN_ROWS:
        out['blockers'].append(
            f'Only {len(df) - n_missing_y} usable rows — at least {MIN_ROWS} are needed to train and evaluate.')
    if yv.nunique() < 2:
        out['blockers'].append(
            'The target column is constant (single value) — nothing can be learned from it.')

    if out['task'] == 'classification':
        counts = yv.value_counts()
        out['classes'] = [{'label': str(k), 'count': int(v)} for k, v in counts.head(12).items()]
        out['nClasses'] = int(counts.size)
        if counts.size > 50:
            out['warnings'].append(
                f'{counts.size} distinct classes — did you mean regression? You can switch the task below.')
        rare = counts[counts < 2]
        if len(rare):
            out['warnings'].append(
                f'{len(rare)} class(es) have a single example ({", ".join(map(str, rare.index[:5]))}) — '
                'they cannot be split into train/test reliably.')
        imb = float(counts.max() / max(counts.min(), 1))
        out['imbalanceRatio'] = round(imb, 2)
        if imb >= 3:
            out['warnings'].append(
                f'Unbalanced classes (majority/minority ratio ≈ {imb:.1f}). The paper reports that PDL '
                'tends to underperform on unbalanced data — class weighting is applied to compensate, '
                'but expect smaller (or negative) gains.')
    else:
        out['targetStats'] = {
            'min': float(yv.min()), 'max': float(yv.max()),
            'mean': float(yv.mean()), 'std': float(yv.std())}

    # feature diagnostics
    feats = [c for c in df.columns if c != tgt]
    missing = df[feats].isna().sum()
    tot_missing = int(missing.sum())
    out['missingCells'] = tot_missing
    out['missingPct'] = round(100.0 * tot_missing / max(1, len(df) * max(1, len(feats))), 2)
    if tot_missing:
        worst = missing.sort_values(ascending=False)
        worst = worst[worst > 0]
        out['warnings'].append(
            f'{tot_missing} missing feature values ({out["missingPct"]}% of cells; worst column: '
            f'"{worst.index[0]}" with {int(worst.iloc[0])}). HistGradientBoosting handles missing values '
            'natively; the other models get median/mode imputation. Note: the paper found PDL to be '
            'sensitive to missing data, since differences with imputed values blur the pairwise signal.')
    non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])
                   and not pd.api.types.is_bool_dtype(df[c])
                   and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if non_numeric:
        out['warnings'].append(
            f'{len(non_numeric)} non-numeric feature(s) ({", ".join(map(str, non_numeric[:6]))}'
            f'{"…" if len(non_numeric) > 6 else ""}) will be ordinal-encoded so the pairwise '
            'difference (a − b) is defined. One-hot encoding may work better for truly nominal data.')

    if len(df) > MAX_TRAIN:
        out['warnings'].append(
            f'Quick-demo limit: your file has {len(df)} rows, but this in-browser PoC trains on a random '
            f'sample of {MAX_TRAIN} rows only. (PDL builds an n² pairwise matrix — the paper also focuses '
            'on datasets below ~2000 rows, where PDL shines.)')

    if S.test_df is not None:
        t = S.test_df
        out['testRows'] = int(len(t))
        missing_cols = [c for c in feats if c not in t.columns]
        if missing_cols:
            out['blockers'].append(
                f'The test file is missing feature column(s): {", ".join(map(str, missing_cols[:8]))}. '
                'Train and test must share the same input columns.')
        if tgt not in t.columns:
            out['warnings'].append(
                f'The test file has no "{tgt}" column, so it cannot be scored. '
                'Scores will be computed on a 20% hold-out split of the training data instead.')
    else:
        out['warnings'].append(
            'No test file uploaded — a stratified 80/20 hold-out split of the training data will be used for scoring.')

    return json.dumps(out)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def _encode_features(df, feats, encoders=None, notes=None):
    """Turn every feature into float32. Returns (X, encoders)."""
    fit = encoders is None
    encoders = encoders if encoders is not None else {}
    cols = {}
    for c in feats:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            cols[c] = s.astype('float64')
        elif pd.api.types.is_datetime64_any_dtype(s):
            cols[c] = s.astype('int64').astype('float64') / 86_400e9  # days
        elif pd.api.types.is_numeric_dtype(s):
            cols[c] = pd.to_numeric(s, errors='coerce').astype('float64')
        else:  # strings / categories -> ordinal codes, NaN preserved
            if fit:
                cats = pd.Index(s.dropna().astype(str).unique())
                encoders[c] = list(cats)
            cats = pd.Index(encoders.get(c, []))
            codes = s.astype(str).where(s.notna(), None).map({v: i for i, v in enumerate(cats)})
            unseen = int((s.notna() & codes.isna()).sum())
            if unseen and notes is not None:
                notes.append(f'Column "{c}": {unseen} unseen categorie(s) in the test file were treated as missing.')
            cols[c] = codes.astype('float64')
    X = pd.DataFrame(cols, index=df.index)
    return X, encoders


def prepare(config_json: str) -> str:
    """config: {target, task}. Builds train/test matrices; returns JSON summary."""
    cfg = json.loads(config_json)
    tgt, task = cfg['target'], cfg['task']
    notes, blockers = [], []
    df = S.train_df.copy()

    df = df[df[tgt].notna()]
    feats = [c for c in df.columns if c != tgt]

    # drop useless columns
    dropped = []
    for c in list(feats):
        s = df[c]
        if s.isna().all():
            dropped.append((c, 'entirely missing'))
        elif s.nunique(dropna=True) <= 1:
            dropped.append((c, 'constant'))
        elif (not pd.api.types.is_numeric_dtype(s)) and s.astype(str).nunique() == len(df) and len(df) > 20:
            dropped.append((c, 'looks like a unique ID'))
    for c, why in dropped:
        feats.remove(c)
    if dropped:
        notes.append('Dropped column(s): ' + ', '.join(f'"{c}" ({why})' for c, why in dropped) + '.')
    if not feats:
        blockers.append('No usable input features remain after cleaning.')
        return json.dumps({'ok': False, 'blockers': blockers, 'notes': notes})

    X_all, encoders = _encode_features(df, feats, notes=notes)
    # columns that became all-NaN after encoding
    dead = [c for c in X_all.columns if X_all[c].notna().sum() == 0]
    if dead:
        X_all = X_all.drop(columns=dead)
        feats = [c for c in feats if c not in dead]
        notes.append(f'Dropped unparseable column(s): {", ".join(dead)}.')
        if not feats:
            blockers.append('No usable input features remain after cleaning.')
            return json.dumps({'ok': False, 'blockers': blockers, 'notes': notes})

    if task == 'classification':
        y_raw = df[tgt].astype(str)
        classes = sorted(y_raw.unique())
        class_to_code = {c: i for i, c in enumerate(classes)}
        y_all = y_raw.map(class_to_code).astype('int64')
        if len(classes) < 2:
            blockers.append('The target has a single class — nothing to classify.')
    else:
        y_all = pd.to_numeric(df[tgt], errors='coerce')
        bad = int(y_all.isna().sum())
        if bad:
            notes.append(f'{bad} target value(s) are not numeric and were dropped (regression task).')
            keep = y_all.notna()
            X_all, y_all = X_all[keep], y_all[keep]
        classes, class_to_code = None, None
        if y_all.nunique() < 2:
            blockers.append('The target column is constant — nothing to regress.')
    if blockers:
        return json.dumps({'ok': False, 'blockers': blockers, 'notes': notes})

    orig_rows = len(X_all)

    # external test set?
    ext_test = False
    X_test = y_test = None
    if S.test_df is not None and tgt in S.test_df.columns:
        tdf = S.test_df[S.test_df[tgt].notna()].copy()
        missing_cols = [c for c in feats if c not in tdf.columns]
        if missing_cols:
            blockers.append('Test file is missing columns: ' + ', '.join(missing_cols[:8]))
            return json.dumps({'ok': False, 'blockers': blockers, 'notes': notes})
        X_test, _ = _encode_features(tdf, feats, encoders=encoders, notes=notes)
        if task == 'classification':
            unknown = sorted(set(tdf[tgt].astype(str)) - set(classes))
            if unknown:
                notes.append(f'Test rows with unseen target class(es) {unknown[:5]} were dropped.')
                tdf = tdf[tdf[tgt].astype(str).isin(classes)]
                X_test = X_test.loc[tdf.index]
            y_test = tdf[tgt].astype(str).map(class_to_code).astype('int64')
        else:
            y_test = pd.to_numeric(tdf[tgt], errors='coerce')
            keep = y_test.notna()
            X_test, y_test = X_test[keep], y_test[keep]
        ext_test = True
        X_train, y_train = X_all, y_all
        if len(X_test) == 0:
            blockers.append('The test file has no scoreable rows.')
            return json.dumps({'ok': False, 'blockers': blockers, 'notes': notes})
    if not ext_test:
        from sklearn.model_selection import train_test_split
        strat = y_all if (task == 'classification' and y_all.value_counts().min() >= 2) else None
        if task == 'classification' and strat is None:
            notes.append('Some classes have a single example — the hold-out split cannot be stratified.')
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=SEED, stratify=strat)
        notes.append(f'Hold-out evaluation: {len(X_train)} train / {len(X_test)} test rows (80/20 split).')

    # ---- quick-demo cap -------------------------------------------------
    sampled = False
    if len(X_train) > MAX_TRAIN:
        if task == 'classification':
            frac = MAX_TRAIN / len(X_train)
            idx = (pd.Series(y_train.values, index=X_train.index)
                   .groupby(y_train.values, group_keys=False)
                   .apply(lambda s: s.sample(max(1, int(round(len(s) * frac))), random_state=SEED)).index)
            idx = idx[:MAX_TRAIN]
        else:
            idx = X_train.sample(MAX_TRAIN, random_state=SEED).index
        X_train, y_train = X_train.loc[idx], y_train.loc[idx]
        sampled = True
        notes.append(f'Quick-demo cap: training on a random sample of {len(X_train)} of the {orig_rows} rows.')

    # ---- pairwise memory cap --------------------------------------------
    n_feat = X_train.shape[1]
    pdl_cap = int(np.sqrt(PAIR_BUDGET / max(1, 3 * n_feat)))
    pdl_cap = max(80, min(pdl_cap, MAX_TRAIN))
    if len(X_train) > pdl_cap:
        if task == 'classification':
            frac = pdl_cap / len(X_train)
            idx = (pd.Series(y_train.values, index=X_train.index)
                   .groupby(y_train.values, group_keys=False)
                   .apply(lambda s: s.sample(max(1, int(round(len(s) * frac))), random_state=SEED)).index)
            idx = idx[:pdl_cap]
        else:
            idx = X_train.sample(pdl_cap, random_state=SEED).index
        X_train, y_train = X_train.loc[idx], y_train.loc[idx]
        notes.append(
            f'With {n_feat} features, the pairwise matrix ({len(X_train)}² pairs × {3 * n_feat} columns) must fit '
            f'in browser memory — all models therefore train on the same {len(X_train)} rows (fair comparison).')
        sampled = True

    X_train = X_train.reset_index(drop=True).astype('float32')
    X_test = X_test.reset_index(drop=True).astype('float32')
    y_train = pd.Series(np.asarray(y_train), name='target').reset_index(drop=True)
    y_test = pd.Series(np.asarray(y_test), name='target').reset_index(drop=True)

    # imputation (median) for models that can't handle NaN
    has_missing = bool(X_train.isna().any().any() or X_test.isna().any().any())
    medians = X_train.median(numeric_only=True).fillna(0.0)
    X_train_imp = X_train.fillna(medians).astype('float32')
    X_test_imp = X_test.fillna(medians).astype('float32')
    if has_missing:
        notes.append('Median imputation applied for DecisionTree/RandomForest/ExtraTrees and all PDL models. '
                     'HistGradientBoosting is trained on the raw data (it supports missing values natively).')

    if task == 'classification':
        counts = pd.Series(y_train).value_counts()
        imbalance = float(counts.max() / max(1, counts.min()))
        n_classes = int(counts.size)
        if n_classes < 2:
            return json.dumps({'ok': False, 'notes': notes,
                               'blockers': ['After sampling/splitting, the training set has a single class.']})
    else:
        imbalance, n_classes = None, None

    S.prep = dict(task=task, target=tgt, feats=feats, encoders=encoders, classes=classes,
                  X_train=X_train, X_test=X_test, X_train_imp=X_train_imp, X_test_imp=X_test_imp,
                  y_train=y_train, y_test=y_test, medians=medians, has_missing=has_missing,
                  orig_rows=orig_rows, ext_test=ext_test, sampled=sampled,
                  imbalance=imbalance, n_classes=n_classes)
    S.models, S.results = {}, {}

    return json.dumps({
        'ok': True, 'notes': notes, 'blockers': [],
        'trainRows': int(len(X_train)), 'testRows': int(len(X_test)),
        'nFeatures': int(X_train.shape[1]), 'task': task,
        'extTest': ext_test, 'hasMissing': has_missing,
        'nClasses': n_classes, 'imbalance': round(imbalance, 2) if imbalance else None,
        'models': [{'id': m['id'], 'name': m['name'], 'kind': m['kind']} for m in _model_list(task)],
    })


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _model_list(task):
    """Ordered fastest-first; baselines before their PDL counterparts."""
    if task == 'classification':
        return [
            dict(id='dt', name='DecisionTree', kind='baseline', base=None),
            dict(id='hgb', name='HistGradientBoosting', kind='baseline', base=None),
            dict(id='et', name='ExtraTrees', kind='baseline', base=None),
            dict(id='rf', name='RandomForest', kind='baseline', base=None),
            dict(id='pdl_dt', name='PDL(DecisionTree)', kind='pdl', base='dt'),
            dict(id='pdl_et', name='PDL(ExtraTrees)', kind='pdl', base='et'),
            dict(id='pdl_rf', name='PDL(RandomForest)', kind='pdl', base='rf'),
        ]
    return [
        dict(id='dt', name='DecisionTree', kind='baseline', base=None),
        dict(id='hgb', name='HistGradientBoosting', kind='baseline', base=None),
        dict(id='et', name='ExtraTrees', kind='baseline', base=None),
        dict(id='rf', name='RandomForest', kind='baseline', base=None),
        dict(id='pdl_dt', name='PDL(DecisionTree)', kind='pdl', base='dt'),
        dict(id='pdl_et', name='PDL(ExtraTrees)', kind='pdl', base='et'),
        dict(id='pdl_rf', name='PDL(RandomForest)', kind='pdl', base='rf'),
    ]


def _make_estimator(mid, task, n_train):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor,
                                  HistGradientBoostingClassifier, HistGradientBoostingRegressor)
    from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

    # modest ensemble sizes: everything runs single-threaded in WebAssembly
    n_estim = 100
    n_estim_pdl = 30 if n_train > 250 else 50
    # The pairwise train set has n² rows; fully-grown forests on it exhaust browser
    # memory, so leaf size scales with the number of pairs to bound the tree count.
    n_pairs = n_train * n_train
    msl = 1 if n_pairs <= 40_000 else 5 if n_pairs <= 120_000 else 10 if n_pairs <= 250_000 else 20
    clf = task == 'classification'
    bal = dict(class_weight='balanced') if clf else {}
    if mid == 'dt':
        return DecisionTreeClassifier(random_state=0, **bal) if clf else DecisionTreeRegressor(random_state=0)
    if mid == 'hgb':
        return (HistGradientBoostingClassifier(random_state=0, **bal) if clf
                else HistGradientBoostingRegressor(random_state=0))
    if mid == 'et':
        return (ExtraTreesClassifier(n_estimators=n_estim, random_state=0, n_jobs=1, **bal) if clf
                else ExtraTreesRegressor(n_estimators=n_estim, random_state=0, n_jobs=1))
    if mid == 'rf':
        return (RandomForestClassifier(n_estimators=n_estim, random_state=0, n_jobs=1, **bal) if clf
                else RandomForestRegressor(n_estimators=n_estim, random_state=0, n_jobs=1))
    if mid.startswith('pdl_'):
        inner = mid.split('_', 1)[1]
        if inner == 'dt':
            base = DecisionTreeClassifier(random_state=0, **bal) if clf else DecisionTreeRegressor(random_state=0)
        elif inner == 'et':
            base = (ExtraTreesClassifier(n_estimators=n_estim_pdl, min_samples_leaf=msl, random_state=0, n_jobs=1, **bal) if clf
                    else ExtraTreesRegressor(n_estimators=n_estim_pdl, min_samples_leaf=msl, random_state=0, n_jobs=1))
        else:
            base = (RandomForestClassifier(n_estimators=n_estim_pdl, min_samples_leaf=msl, random_state=0, n_jobs=1, **bal) if clf
                    else RandomForestRegressor(n_estimators=n_estim_pdl, min_samples_leaf=msl, random_state=0, n_jobs=1))
        return PairwiseDifferenceClassifier(estimator=base) if clf else PairwiseDifferenceRegressor(estimator=base)
    raise ValueError(mid)


def _metrics(task, y_true, y_pred):
    from sklearn import metrics as M
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task == 'classification':
        return {
            'accuracy': round(float(M.accuracy_score(y_true, y_pred)), 4),
            'f1_macro': round(float(M.f1_score(y_true, y_pred, average='macro')), 4),
        }
    return {
        'mae': round(float(M.mean_absolute_error(y_true, y_pred)), 5),
        'rmse': round(float(np.sqrt(M.mean_squared_error(y_true, y_pred))), 5),
        'r2': round(float(M.r2_score(y_true, y_pred)), 4),
    }


def _explain(task, kind, delta, prep):
    """Paper-grounded explanation of why PDL improved / degraded."""
    if kind != 'pdl' or delta is None:
        return []
    n = len(prep['y_train'])
    reasons = []
    better = delta > 0.001
    worse = delta < -0.001
    if better:
        reasons.append(
            f'PDL learns from pairs: {n} training rows become {n}×{n} = {n * n:,} pairwise examples, '
            'acting as a data augmentation — the paper shows this is where PDL gains most on small datasets.')
        if task == 'classification':
            reasons.append('The multiclass problem is reduced to one binary question — "are these two points '
                           'the same class?" — which is statistically easier to learn (Belaid et al., 2024).')
    if worse or (not better and not worse):
        if prep.get('has_missing'):
            reasons.append('Your data contains missing values. The paper found PDL sensitive to missing data: '
                           'pairwise differences computed against imputed values blur the similarity signal.')
        if task == 'classification' and prep.get('imbalance') and prep['imbalance'] >= 3:
            reasons.append(f'Classes are unbalanced (ratio ≈ {prep["imbalance"]:.1f}). The paper reports weaker '
                           'PDL performance on unbalanced datasets, even with class weighting and prior correction.')
        if prep.get('orig_rows', 0) > 2000:
            reasons.append('The original dataset exceeds 2000 rows. The paper focuses on datasets below ~2000 '
                           'samples; with abundant data the baseline catches up and the n² pairing cost stops paying off.')
        if not reasons:
            reasons.append('PDL gains are dataset-dependent: they are largest on small, balanced, fully observed '
                           'tabular data with tree-based learners (per the Discovery Science 2024 benchmark on 99 datasets).')
    if worse:
        reasons.append('Tip: PDL only pairs well with non-parametric tree learners (DecisionTree, RandomForest, '
                       'ExtraTrees). With parametric models (linear, SVM, kNN, MLP) the paper observed degradation.')
    return reasons


def train_all() -> str:
    """Train every model, emitting a progress event after each; never stops on a single failure."""
    p = S.prep
    assert p is not None, 'prepare() must be called first'
    task = p['task']
    results = []
    for m in _model_list(task):
        _emit({'event': 'start', 'id': m['id'], 'name': m['name'], 'kind': m['kind']})
        res = {'id': m['id'], 'name': m['name'], 'kind': m['kind'], 'base': m['base']}
        t0 = time.time()
        try:
            est = _make_estimator(m['id'], task, len(p['y_train']))
            use_raw = m['id'] == 'hgb'  # HGB consumes NaNs natively
            Xtr = p['X_train'] if use_raw else p['X_train_imp']
            Xte = p['X_test'] if use_raw else p['X_test_imp']
            est.fit(Xtr, p['y_train'])
            y_pred = est.predict(Xte)
            res['metrics'] = _metrics(task, p['y_test'], y_pred)
            res['seconds'] = round(time.time() - t0, 2)
            res['status'] = 'ok'
            S.models[m['id']] = est
            # delta vs. base learner on the primary metric
            if m['kind'] == 'pdl' and m['base'] in S.results and S.results[m['base']].get('status') == 'ok':
                bm = S.results[m['base']]['metrics']
                if task == 'classification':
                    delta = res['metrics']['f1_macro'] - bm['f1_macro']
                else:
                    delta = bm['mae'] - res['metrics']['mae']  # positive = PDL better
                res['delta'] = round(float(delta), 4)
                res['explanation'] = _explain(task, m['kind'], delta, p)
        except MemoryError:
            res['status'] = 'error'
            res['seconds'] = round(time.time() - t0, 2)
            res['error'] = ('Out of browser memory while building the pairwise matrix. '
                            'Try fewer rows or fewer features.')
        except Exception as e:
            res['status'] = 'error'
            res['seconds'] = round(time.time() - t0, 2)
            res['error'] = f'{type(e).__name__}: {e}'
            res['trace'] = traceback.format_exc(limit=3)
        S.results[m['id']] = res
        results.append(res)
        _emit({'event': 'done', **res})

    ok = [r for r in results if r['status'] == 'ok']
    best = None
    if ok:
        if task == 'classification':
            best = max(ok, key=lambda r: r['metrics']['f1_macro'])['id']
        else:
            best = min(ok, key=lambda r: r['metrics']['mae'])['id']
    return json.dumps({'event': 'all_done', 'bestId': best, 'results': results,
                       'nOk': len(ok), 'nFail': len(results) - len(ok)})


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
def get_model_pickle(model_id: str):
    import sklearn
    import pdll as _pdll
    p = S.prep
    est = S.models[model_id]
    bundle = {
        'model': est,
        'model_name': dict((m['id'], m['name']) for m in _model_list(p['task']))[model_id],
        'task': p['task'],
        'target': p['target'],
        'feature_names': list(p['feats']),
        'category_encoders': p['encoders'],       # {column: [category order]} -> ordinal codes
        'impute_medians': {k: float(v) for k, v in p['medians'].items()},
        'class_labels': p['classes'],             # code i -> label (classification only)
        'trained_on_rows': int(len(p['y_train'])),
        'sklearn_version': sklearn.__version__,
        'pdll_version': getattr(_pdll, '__version__', 'dev'),
        'note': 'Trained fully in-browser by the PDL live demo (https://github.com/Karim-53/pdll).',
    }
    return pickle.dumps(bundle)


def get_load_snippet(model_id: str) -> str:
    import sklearn
    p = S.prep
    lines = [
        '# pip install pdll scikit-learn=={}'.format(sklearn.__version__),
        'import pickle, pandas as pd',
        '',
        "bundle = pickle.load(open('pdl_demo_{}.pkl', 'rb'))".format(model_id),
        "model = bundle['model']",
        '',
        '# prepare new data exactly like the demo did:',
        "df = df[bundle['feature_names']].copy()",
        "for col, cats in bundle['category_encoders'].items():",
        '    df[col] = df[col].astype(str).map({c: i for i, c in enumerate(cats)})',
        "df = df.fillna(bundle['impute_medians']).astype('float32')",
        '',
        'pred = model.predict(df)',
    ]
    if p['task'] == 'classification':
        lines.append("labels = [bundle['class_labels'][int(i)] for i in pred]  # back to original labels")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Local smoke test:  python trainer.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find ./pdll
    set_progress_callback(lambda s: print('  progress:', json.loads(s).get('event'),
                                          json.loads(s).get('name', ''),
                                          json.loads(s).get('metrics', json.loads(s).get('error', ''))))
    for demo in ('iris', 'blobs', 'diabetes'):
        print(f'=== {demo} ===')
        a = json.loads(load_demo(demo))
        print('analysis:', {k: a[k] for k in ('task', 'nRows', 'target') if k in a})
        prep = json.loads(prepare(json.dumps({'target': a['target'], 'task': a['task']})))
        print('prepared:', {k: prep[k] for k in ('trainRows', 'testRows', 'nFeatures') if k in prep})
        assert prep['ok'], prep
        summary = json.loads(train_all())
        print('best:', summary['bestId'], '| ok:', summary['nOk'], 'fail:', summary['nFail'])
        blob = get_model_pickle(summary['bestId'])
        print('pickle bytes:', len(blob))
        assert summary['nFail'] == 0, [r for r in summary['results'] if r['status'] != 'ok']
    print('ALL GOOD')
