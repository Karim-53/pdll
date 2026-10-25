"""Check that ``run_one.load`` gives exactly TabMini's data: same rows in the same order, same 0/1 labels.

    python check_data.py

Compares with the data snapshot in TabMini's repository (``plotting/data/<size bin>/<name>/X.csv, y.csv`` at commit
385961e), which TabMini saved after its own shuffling and label encoding. Identical data and labels give identical
folds, since both sides use ``StratifiedKFold(3)`` (what TabMini's ``cv=3`` means for a classifier). No model is fitted.
"""
import io
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from run_all import DATASETS
from run_one import load

SNAPSHOT = 'https://raw.githubusercontent.com/RicardoKnauer/TabMini/385961e/plotting/data/{bin}/{name}/{file}'


def snapshot(name):
    for size_bin in range(1, 6):
        try:
            X = pd.read_csv(io.BytesIO(urllib.request.urlopen(SNAPSHOT.format(bin=size_bin, name=name, file='X.csv')).read()))
        except urllib.error.HTTPError:
            continue
        y = pd.read_csv(io.BytesIO(urllib.request.urlopen(SNAPSHOT.format(bin=size_bin, name=name, file='y.csv')).read()))
        return X, y.iloc[:, 0].to_numpy()
    raise FileNotFoundError(name)


def main():
    different = []
    for name in DATASETS:
        X, y = load(name)
        X_ref, y_ref = snapshot(name)
        same = (X.shape == X_ref.shape and np.allclose(X.to_numpy(float), X_ref.to_numpy(float), equal_nan=True)
                and np.array_equal(y, y_ref))
        if not same:
            different.append(name)
    print(f'{len(DATASETS) - len(different)} of {len(DATASETS)} datasets identical to TabMini\'s snapshot'
          + (f'; different: {different}' if different else ''))


if __name__ == '__main__':
    main()
