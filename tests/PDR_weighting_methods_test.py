import sys
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

sys.path.extend('../')
from pdll import PairwiseDifferenceRegressor

def test_weighted_pdr():
    # Set the random seed for reproducibility:
    np.random.seed(53)

    # Define the number of data points and features:
    n_samples = 50
    n_validation_samples = 10
    n_test_samples = 10
    n_features = 8
    n_classes = 3

    # Generate random data with 8 features, 50 points, and 3 classes:
    X, y = make_regression(n_samples=n_samples+n_validation_samples+n_test_samples, n_features=n_features, random_state=53)

    # Random split:
    X_train = X[:n_samples]
    X_validation = X[n_samples:n_samples+n_validation_samples]
    X_test = X[n_samples+n_validation_samples:]
    y_train = y[:n_samples]
    y_validation = y[n_samples:n_samples+n_validation_samples]
    y_test = y[n_samples+n_validation_samples:]

    # Train the PairwiseDifferenceRegressor model:
    pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())

    # Fit the model:
    pdr.fit(X_train, y_train)

    # Print the score:
    print('train score:', pdr.score(X_train, y_train))
    print('test score:', pdr.score(X_test, y_test))

    # Now try out all the weighting methods to see if any of them improve the score:
    for method in pdr._name_to_method_mapping.keys():
        print(f"\n\nScore for {method}:")
        pdr.learn_anchor_weights(X_validation, y_validation, method=method)
        print('train score:', pdr.score(X_train, y_train))
        test_score = pdr.score(X_test, y_test)
        print('test score:', test_score)
        assert test_score < .7
