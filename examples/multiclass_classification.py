"""Quick test."""
import sys
sys.path.extend('../')
from pdll import PairwiseDifferenceClassifier

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(53)

    # Define the number of data points and features
    n_samples = 10
    n_features = 2
    n_classes = 3

    # Generate random data with 2 features, 10 points, and 3 classes
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=0)

    pdc = PairwiseDifferenceClassifier(estimator=RandomForestClassifier())
    pdc.fit(X, y)
    print('score:', pdc.score(X, y))

    y_pred = pdc.predict(X)
    proba_pred = pdc.predict_proba(X)

    assert pdc.score(X, y) == 1.0
