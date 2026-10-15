"""Registry of real-world datasets used to validate pdll before a release.

Every loader returns the data as a user would get it (raw DataFrame, raw target), so the campaign also tests input
handling: strings, categories, missing values, odd dtypes. Sources:

- OpenML-CC18 (72 classification datasets) and OpenML-CTR23 (35 regression datasets), the curated OpenML suites;
- datasets that originate on Kaggle, through their OpenML mirrors (no Kaggle account needed);
- the datasets bundled with or fetched by scikit-learn.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Kaggle datasets mirrored on OpenML: (OpenML id, short name, task, target or None for the OpenML default)
KAGGLE_ON_OPENML = [
    (40945, 'kaggle_titanic_raw', 'classification', 'survived'),
    (42165, 'kaggle_house_prices', 'regression', 'SalePrice'),
    (43595, 'kaggle_loan_prediction', 'classification', None),
    (42178, 'kaggle_telco_churn', 'classification', None),
    (46612, 'kaggle_heart_failure', 'classification', 'death_event'),
    (43682, 'kaggle_heart_disease_comprehensive', 'classification', None),
    (46931, 'kaggle_insurance_expenses', 'regression', None),
    (46944, 'kaggle_mobile_price', 'classification', None),
    (46960, 'kaggle_students_dropout', 'classification', None),
    (46925, 'kaggle_employee_attrition', 'classification', None),
    (44229, 'kaggle_churn_telco_europa', 'classification', None),
    (1466, 'kaggle_cardiotocography', 'classification', None),
    (1597, 'kaggle_creditcard_fraud', 'classification', None),
]

# Columns that are identifiers in the raw data: users drop them after a first look, so do we.
ID_COLUMNS = {'kaggle_titanic_raw': ['name', 'ticket', 'boat', 'body', 'home.dest'], 'kaggle_house_prices': ['Id'],
              'zindi_autoinland_vehicle_insurance': ['ID'],
              'kaggle_loan_prediction': ['Loan_ID'], 'kaggle_telco_churn': ['customerID']}


def registry():
    """List of dataset specs: dict(name, source, task, loader args)."""
    specs = []
    suites = pd.read_csv(os.path.join(HERE, 'openml_suites.csv'))
    for row in suites.itertuples():
        task = 'classification' if row.suite == 'CC18' else 'regression'
        specs.append({'name': f'openml_{row.name}', 'source': f'OpenML-{row.suite}', 'task': task, 'openml_id': int(row.did),
                      'openml_task': int(row.tid), 'rows': int(row.NumberOfInstances), 'features': int(row.NumberOfFeatures) - 1})
    for did, name, task, target in KAGGLE_ON_OPENML:
        specs.append({'name': name, 'source': 'Kaggle (OpenML mirror)', 'task': task, 'openml_id': did, 'target': target})
    specs.append({'name': 'zindi_autoinland_vehicle_insurance', 'source': 'Zindi (private copy, not redistributed)',
                  'task': 'classification', 'github_csv': ('Karim-53/Vehicle-Insurance-Claim-Challenge', 'data/Train.csv'),
                  'target': 'target'})
    for name, task in [('iris', 'classification'), ('wine', 'classification'), ('breast_cancer', 'classification'),
                       ('digits', 'classification'), ('olivetti_faces', 'classification'), ('diabetes', 'regression'),
                       ('covtype', 'classification'), ('kddcup99_SA', 'classification'), ('linnerud', 'regression'),
                       ('20newsgroups_vectorized', 'classification')]:
        specs.append({'name': f'sklearn_{name}', 'source': 'scikit-learn', 'task': task, 'sklearn': name})
    return specs


def load(spec):
    """Return (X, y) exactly as the source provides them, minus identifier columns."""
    if 'github_csv' in spec:
        X = _read_github_csv(*spec['github_csv'])
        y = X.pop(spec['target'])
    elif 'openml_id' in spec:
        import openml
        dataset = openml.datasets.get_dataset(spec['openml_id'], download_data=True, download_qualities=False,
                                              download_features_meta_data=False)
        target = spec.get('target') or dataset.default_target_attribute
        X, y, _, _ = dataset.get_data(target=target, dataset_format='dataframe')
    else:
        X, y = _load_sklearn(spec['sklearn'])
    drop = [c for c in ID_COLUMNS.get(spec['name'], []) if c in getattr(X, 'columns', [])]
    if drop:
        X = X.drop(columns=drop)
    return X, y


def _read_github_csv(repo, path):
    """Read a CSV from a GitHub repo through the local gh session (works for private repos), cached on disk."""
    import subprocess
    cache = os.path.join(os.path.expanduser('~'), '.cache', 'pdll_benchmark', repo.replace('/', '_'), path.replace('/', '_'))
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        content = subprocess.run(['gh', 'api', f'repos/{repo}/contents/{path}', '-H', 'Accept: application/vnd.github.raw'],
                                 capture_output=True, check=True).stdout
        with open(cache, 'wb') as f:
            f.write(content)
    return pd.read_csv(cache)


def _load_sklearn(name):
    from sklearn import datasets
    if name == 'kddcup99_SA':
        bunch = datasets.fetch_kddcup99(subset='SA', as_frame=True, percent10=True)
        return bunch.data, bunch.target
    if name == 'covtype':
        return datasets.fetch_covtype(return_X_y=True, as_frame=True)
    if name == 'olivetti_faces':
        X, y = datasets.fetch_olivetti_faces(return_X_y=True)
        return X, y
    if name == '20newsgroups_vectorized':
        return datasets.fetch_20newsgroups_vectorized(return_X_y=True)  # sparse: pdll must refuse it clearly
    if name == 'linnerud':
        return datasets.load_linnerud(return_X_y=True, as_frame=True)  # 3 targets: pdll must refuse it clearly
    loader = getattr(datasets, f'load_{name}')
    return loader(return_X_y=True, as_frame=True)


if __name__ == '__main__':
    specs = registry()
    print(len(specs), 'datasets:', pd.Series([s['source'] for s in specs]).value_counts().to_dict())
    print(np.unique([s['task'] for s in specs], return_counts=True))
