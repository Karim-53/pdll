"""Does balanced sample weighting fix PDL's minority-class under-prediction on imbalanced data?

    python imbalance_probe.py openml_pc4 openml_ilpd ...
"""
import json
import sys
import warnings

import numpy as np
import openml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

import datasets
from pdll import PairwiseDifferenceClassifier

warnings.simplefilter('ignore')
for name in sys.argv[1:]:
    spec = next(s for s in datasets.registry() if s['name'] == name)
    X, y = datasets.load(spec)
    keep = ~y.isna()
    X, y = X[keep], y[keep]
    if 'openml_task' in spec:
        train, test = openml.tasks.get_task(spec['openml_task'], download_splits=True, download_data=False,
                                            download_qualities=False).get_train_test_split_indices(fold=0, repeat=0)
    else:
        order = np.random.RandomState(0).permutation(len(y))
        train, test = order[:int(0.75 * len(y))], order[int(0.75 * len(y)):][:2000]
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    minority = float(y_train.value_counts(normalize=True).min())
    out = {'dataset': name, 'minority': round(minority, 3)}
    for label, weights in (('plain', None), ('balanced', compute_sample_weight('balanced', y_train))):
        model = PairwiseDifferenceClassifier(random_state=0).fit(X_train, y_train, sample_weight=weights)
        proba = model.predict_proba(X_test)
        out[f'{label}_f1'] = round(f1_score(y_test, model.predict(X_test), average='macro'), 3)
        if len(model.classes_) == 2:
            out[f'{label}_auc'] = round(roc_auc_score(np.asarray(y_test) == model.classes_[1], proba[:, 1]), 3)
    print(json.dumps(out), flush=True)
