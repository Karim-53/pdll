"""Quick test."""
import sys

sys.path.extend('../')
from pdll import PairwiseDifferenceRegressor

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression


def test_regression_task():
    # Set the random seed for reproducibility
    np.random.seed(53)

    X, y = make_regression(n_samples=50, n_features=8, random_state=53)

    pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())
    pdr.fit(X, y)
    print('score:', pdr.score(X, y))

    y_pred = pdr.predict(X)

    assert pdr.score(X, y) == 1.0


if __name__ == "__main__":
    test_regression_task()
