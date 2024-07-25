import unittest

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from pdll import PairwiseDifferenceRegressor


def load_fake_data(n_samples, n_validation_samples, n_test_samples, n_features, noise_level=0.):
    # Set the random seed for reproducibility:
    np.random.seed(42)

    # Generate random data with 8 features, 50 points, and 3 classes:
    X, y = make_regression(n_samples=n_samples + n_validation_samples + n_test_samples, n_features=n_features,
                           random_state=53, noise=noise_level)

    # Random split:
    X_train, X_validation, X_test = (X[:n_samples],
                                     X[n_samples:n_samples + n_validation_samples],
                                     X[n_samples + n_validation_samples:])
    y_train, y_validation, y_test = (y[:n_samples],
                                     y[n_samples:n_samples + n_validation_samples],
                                     y[n_samples + n_validation_samples:])

    # To pandas dataframe:
    train_df = pd.DataFrame(X_train)
    y_train_df = pd.Series(y_train)
    validation_df = pd.DataFrame(X_validation)
    y_validation_df = pd.Series(y_validation)
    test_df = pd.DataFrame(X_test)
    y_test_df = pd.Series(y_test)

    return (train_df, validation_df, test_df,
            y_train_df, y_validation_df, y_test_df)


def score_weighting_stability1(weighting_method):
    fake_data = load_fake_data(
        n_samples=50,
        n_validation_samples=10,
        n_test_samples=10,
        n_features=8,
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = fake_data

    pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())
    pdr.fit(X_train, y_train)

    pdr.learn_anchor_weights(X_validation, y_validation, method=weighting_method)
    return pdr.score(X_test, y_test)


def score_weighting_stability2(weighting_method):
    fake_data = load_fake_data(
        n_samples=150,
        n_validation_samples=15,
        n_test_samples=200,
        n_features=15,
        noise_level=0.5,
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = fake_data

    pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())
    pdr.fit(X_train, y_train)

    pdr.learn_anchor_weights(X_validation, y_validation, method=weighting_method)
    return pdr.score(X_test, y_test)


def score_permutation_distance(weighting_method):
    fake_data = load_fake_data(
        n_samples=50,
        n_validation_samples=50,
        n_test_samples=50,
        n_features=8,
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = fake_data

    pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())
    pdr.fit(X_train, y_train)

    pdr.learn_anchor_weights(X_validation, y_validation, method=weighting_method)

    # Get weights and get the argsort of the weights:
    weights = pdr.sample_weight_
    weights = [w for w in weights]
    sorted_weights = np.argsort(weights)

    # Get the error per weight and get the argsort of that:
    error_per_anchor = [-e for e in pdr._error(X_val=X_validation, y_val=y_validation)]
    sorted_errors = np.argsort(error_per_anchor)

    # Sort [1, ..., n] by the argsort of the weights and errors:
    sorted_weights = [sorted_weights[i] for i in range(len(sorted_weights))]
    sorted_errors = [sorted_errors[i] for i in range(len(sorted_errors))]
    # import matplotlib.pyplot as plt
    # plt.plot(sorted_errors, sorted_weights)
    # plt.title(f"Permutation distance for {weighting_method}")
    # plt.show()
    # Check if the ordering of the weights is similar to the ordering of the errors:
    # Kendall's tau is a correlation measure for ordinal data.
    # -1 indicates a strong disagreement and 1 indicates a strong agreement.
    # So generally, we would expect the correlation to be positive.
    return scipy.stats.kendalltau(sorted_weights, sorted_errors).correlation


class WeightedPDRTestPermutationDistance(unittest.TestCase):
    """
    Check the ordering of the produced weight and make sure
    the ordering is similar to the ordering of the error.
    """

    def test_permutation_distance_optimize(self):
        # We expect correlation, but maybe not so strong
        test_score = score_permutation_distance('Optimize')
        self.assertGreater(test_score, -0.1, "Permutation distance is too high!")

    def test_permutation_distance_kld(self):
        # We expect correlation, but maybe not so strong
        test_score = score_permutation_distance('KLD')
        self.assertGreater(test_score, -0.1, "Permutation distance is too high!")

    def test_permutation_distance_l1(self):
        # We expect correlation, but maybe not so strong
        test_score = score_permutation_distance('L1')
        self.assertGreater(test_score, -0.1, "Permutation distance is too high!")

    def test_permutation_distance_l2(self):
        # We expect correlation, but maybe not so strong
        test_score = score_permutation_distance('L2')
        self.assertGreater(test_score, -0.15, "Permutation distance is too high!")

    def test_permutation_distance_elasticnet(self):
        # We expect correlation, but maybe not so strong
        test_score = score_permutation_distance('L1L2')
        self.assertGreater(test_score, -0.1, "Permutation distance is too high!")

    def test_permutation_distance_extreme_weight_pruning(self):
        # We expect correlation, but maybe not so strong
        test_score = score_permutation_distance('ExtremeWeightPruning')
        self.assertGreater(test_score, -0.1, "Permutation distance is too high!")

    def test_permutation_distance_negative_error(self):
        # For error-based methods, correlation should very high
        test_score = score_permutation_distance('NegativeError')
        self.assertGreater(test_score, 0.9, "Permutation distance is too high!")

    def test_permutation_distance_inverse_error(self):
        # For error-based methods, correlation should very high
        test_score = score_permutation_distance('InverseError')
        self.assertGreater(test_score, 0.9, "Permutation distance is too high!")

    def test_sample_weight_ordered_votes_from_weights(self):
        np.testing.assert_array_equal(PairwiseDifferenceRegressor._sample_weight_ordered_votes_from_weights(np.array([0.1, 0.2, 0.7])), np.array([1 / 6, 2 / 6, 3 / 6]))
        np.testing.assert_array_equal(PairwiseDifferenceRegressor._sample_weight_ordered_votes_from_weights(np.array([0.1, 0.7, 0.2])), np.array([1 / 6, 3 / 6, 2 / 6]))
        np.testing.assert_array_equal(PairwiseDifferenceRegressor._sample_weight_ordered_votes_from_weights(np.array([0.5, 0.3, 0])), np.array([3 / 6, 2 / 6, 1 / 6]))
        np.testing.assert_array_equal(PairwiseDifferenceRegressor._sample_weight_ordered_votes_from_weights(np.array([1, 0])), np.array([2 / 3, 1 / 3]))
        np.testing.assert_array_equal(PairwiseDifferenceRegressor._sample_weight_ordered_votes_from_weights(np.array([1,])), np.array([1]))

    def test_permutation_distance_ordered_voting(self):
        # For error-based methods, correlation should be very high
        test_score = score_permutation_distance('OrderedVoting')
        self.assertGreater(test_score, 0.9, "Permutation distance is too high!")


class WeightedPDRTestStability(unittest.TestCase):
    """ Make sure all weighting methods run without crashing. """

    def score_weighting_stability_optimize(self):
        test_score = score_weighting_stability1('Optimize')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('Optimize')
        self.assertLess(test_score2, .9, "Test score is too high!")

    def score_weighting_stability_kld(self):
        test_score = score_weighting_stability1('KLD')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('KLD')
        self.assertLess(test_score2, .9, "Test score is too high!")

    def score_weighting_stability_l1(self):
        test_score = score_weighting_stability1('L1')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('L1')
        self.assertLess(test_score2, .9, "Test score is too high!")

    def score_weighting_stability_l2(self):
        test_score = score_weighting_stability1('L2')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('L2')
        self.assertLess(test_score2, .9, "Test score is too high!")

    def score_weighting_stability_elasticnet(self):
        test_score = score_weighting_stability1('L1L2')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('L1L2')
        self.assertLess(test_score2, .9, "Test score is too high!")

    def score_weighting_stability_extreme_weight_pruning(self):
        test_score = score_weighting_stability1('ExtremeWeightPruning')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('ExtremeWeightPruning')
        self.assertLess(test_score2, .9, "Test score is too high!")

    def score_weighting_stability_negative_error(self):
        test_score = score_weighting_stability1('NegativeError')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('negative error')
        self.assertLess(test_score2, .95, "Test score is too high!")

    def score_weighting_stability_inverse_error(self):
        test_score = score_weighting_stability1('InverseError')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('InverseError')
        self.assertLess(test_score2, .95, "Test score is too high!")

    def score_weighting_stability_ordered_voting(self):
        test_score = score_weighting_stability1('OrderedVoting')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('OrderedVoting')
        self.assertLess(test_score2, .95, "Test score is too high!")

    def score_weighting_stability_KMeansClusterCenters(self):
        test_score = score_weighting_stability1('KMeansclustercenters')
        self.assertLess(test_score, .7, "Test score is too high!")

        test_score2 = score_weighting_stability2('KMeansclustercenters')
        self.assertLess(test_score2, .95, "Test score is too high!")
