"""Quick test."""
import sys


sys.path.extend('../')
from pdll import PairwiseDifferenceRegressor

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression


def test_weighted_regression_task():
    # todo improve this test
    np.random.seed(42)

    X_train, y_train = make_regression(n_samples=50, n_features=8, noise=100., random_state=53)
    X_val, y_val = make_regression(n_samples=50, n_features=8, noise=.5, random_state=53)
    X_test, y_test = make_regression(n_samples=50, n_features=8, noise=.5, random_state=53)
    pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())
    pdr.fit(X_train, y_train)
    # print('score:', pdr.score(X_test, y_test))
    pdr.learn_anchor_weights(X_val, y_val, method='KLD')
    # print(pdr.sample_weight_)
    # print('score:', pdr.score(X_test, y_test))

    # y_pred = pdr.predict(X)
    # assert pdr.score(X, y) == 1.0


if __name__ == "__main__":
    test_weighted_regression_task()
