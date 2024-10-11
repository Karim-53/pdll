"""Pairwise Difference Learning meta-estimator."""
import functools
import warnings
from typing import Iterable

# Author: Mohamed Karim Belaid <karim.belaid@idiada.com> or <extern.karim.belaid@porsche.de>
# License: Apache-2.0 clause

import numpy as np
import pandas as pd
import sklearn.base
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from scipy.special import softmax
from pandas.core.dtypes.common import is_unsigned_integer_dtype


# todo Developing scikit-learn estimators: https://scikit-learn.org/stable/developers/develop.html    and this for common term    https://scikit-learn.org/stable/glossary.html
# todo follow this https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html#sphx-glr-auto-examples-developing-estimators-sklearn-is-fitted-py
# todo PairwiseDifference class that can detect if it is classifiaction or regression if it implement predict proba
# todo change assert to raise error

class PairwiseDifferenceBase(sklearn.base.BaseEstimator):
    """
    Base class for Pairwise Difference Learning.
    """

    @staticmethod
    def pair_input(X1, X2):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        X_pair = X1.merge(X2, how="cross")
        x1_pair = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_diff' for column in X1.columns})
        x2_pair = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_diff' for column in X1.columns})

        try:
            calculate_difference = x1_pair - x2_pair
        except:
            raise ValueError("PairwiseDifference: The input data is not compatible with the subtraction operation. Either transform all data to numeric features or use a ColumnTransformer to transform the data.")
        # It means that the input data is not compatible with the subtraction operation.
        # Simply turn all your data into numbers

        X_pair = pd.concat([X_pair, calculate_difference], axis='columns')
        # Symmetric
        x2_pair_sym = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_y' for column in X1.columns})
        x1_pair_sym = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_x' for column in X1.columns})
        X_pair_sym = pd.concat([x1_pair_sym, x2_pair_sym, x2_pair - x1_pair], axis='columns')

        return X_pair, X_pair_sym

    @staticmethod
    def pair_output(y1: pd.Series, y2: pd.Series) -> pd.Series:
        """For regresion. beware this is different from regression this is b-a not a-b"""
        y_pair = pd.DataFrame(y1).merge(y2, how="cross")
        y_pair_diff = y_pair.iloc[:, 1] - y_pair.iloc[:, 0]
        return y_pair_diff

    @staticmethod
    def pair_output_difference(y1: pd.Series, y2: pd.Series, nb_classes: int) -> pd.Series:
        """For MultiClassClassification base on difference only"""
        y_pair = pd.DataFrame(y1).merge(y2, how="cross")
        y_pair_diff = (y_pair.iloc[:, 1] != y_pair.iloc[:, 0]).astype(int)
        assert y_pair_diff.nunique() <= 2, f'should only be 0s and 1s {y_pair_diff.unique()}'
        return y_pair_diff

    @staticmethod
    def get_pair_feature_names(features: list) -> list:
        """ Get the new name of features after pairing points. """
        return [f'{name}_x' for name in features] + [f'{name}_y' for name in features]

    @staticmethod
    def check_input(X: pd.DataFrame) -> None:
        # todo use https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y
        if X is None:
            raise ValueError('X cannot be None')
        if len(X.shape) != 2:
            raise ValueError('X must be 2D')
        if isinstance(X, pd.DataFrame):
            if 'uint' in str(X.dtypes):
                raise ValueError(f'X cannot have unsigned integers (uint)\n{X.dtypes}')
            # check that all dtypes are numeric
            if any( not np.issubdtype(dtype, np.number) for dtype in X.dtypes):
                raise ValueError(f'X must have numeric dtypes like float32 float64 int... Current dtypes are:\n{X.dtypes}\n You can manually convert/one-hot-encode the features or use a predefined transformer. See https://github.com/Karim-53/pdll/tree/main/examples/any_datatype_train_test.py')
        elif isinstance(X, np.ndarray):
            if is_unsigned_integer_dtype(X):
                raise ValueError(f'X cannot have unsigned integers (uint). Current dtypes:\n{X.dtype}')
            # check that all dtypes are numeric
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError(f'X must have numeric dtypes\n{X.dtype}')

    @staticmethod
    def check_output(y: pd.Series) -> None:
        assert y is not None
        assert isinstance(y, pd.Series)
        assert 'uint' not in str(y.dtype), y.dtype
        assert isinstance(y, pd.Series) or y.shape[1] == 1, f"Didn't expect more than one output {y.shape}"
        assert y.nunique() > 1, y.nunique()
        if y.name is None:
            # just put any name to the output to avoid a bug later
            y.name = 'output'

    @staticmethod
    def check_estimator(estimator, expecting_classifier=False, expecting_regressor=False) -> None:
        try:
            # todo check if it has the fit method
            if isinstance(estimator, sklearn.base.BaseEstimator):
                if expecting_regressor and not sklearn.base.is_regressor(estimator):
                    warnings.warn('estimator must be a regressor.')
                if expecting_classifier:
                    if not sklearn.base.is_classifier(estimator):
                        warnings.warn('estimator must be a classifier.')
                    import inspect
                    if 'class_weight' in inspect.signature(estimator.__class__.__init__).parameters and (estimator.class_weight is None or estimator.class_weight != 'balanced'):
                        warnings.warn('For better performance, estimator should have class_weight="balanced".')

                from sklearn.neural_network import MLPClassifier, MLPRegressor
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
                from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier, LinearRegression, Lasso, Ridge, ElasticNet
                from sklearn.svm import LinearSVC, SVC, SVR
                from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

                incompatible_estimators = (MLPClassifier, KNeighborsClassifier, GaussianNB, BernoulliNB, SGDClassifier, LinearSVC, LogisticRegression,
                                           RidgeClassifier, GaussianProcessClassifier, SVC, MultinomialNB, LinearRegression, MLPRegressor,
                                           KNeighborsRegressor, GaussianProcessRegressor, Ridge, Lasso, ElasticNet, SVR)
                if isinstance(estimator, incompatible_estimators):
                    warnings.warn('PDL is not compatible with base estimators of type parametric models, i.e., even if the code works there is low chance of improvement compared to using directly that estimator. To obtain an improvement, it is better to use a tree-based model like: ExtraTrees, RandomForest, DecisionTree, Bagging, etc.')
            else:
                warnings.warn('Make sure the estimator has a .fit() and .predict() methods.')
        except:
            return

    @staticmethod
    def check_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> None:
        if sample_weight is None:
            pass
        elif isinstance(sample_weight, pd.Series):
            # check
            if len(sample_weight) != len(y_train):
                raise ValueError(
                    f'sample_weight size {len(sample_weight)} should be equal to the train size {len(y_train)}')
            if not sample_weight.index.equals(y_train.index):
                raise ValueError(
                    f'sample_weight and y_train must have the same index\n{sample_weight.index}\n{y_train.index}')
            if all(sample_weight.fillna(0) <= 0):
                raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')

            # norm
            class_sums = np.bincount(y_train, sample_weight)
            sample_weight = sample_weight / class_sums[y_train.astype(int)]
        else:
            raise NotImplementedError()

    @staticmethod
    def correct_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> pd.Series:
        if sample_weight is not None:
            sample_weight = sample_weight / sum(sample_weight)
            # norm
            # class_sums = np.bincount(y_train, sample_weight)
            # sample_weight = sample_weight / class_sums[y_train.astype(int)]

        #     # if sample_weight.min() < 0:  # dolla weight change : improvement +0.0032 bof
        #     #     sample_weight = sample_weight - sample_weight.min()
        return sample_weight

    @staticmethod
    def predict(y_prob, input_type=pd.DataFrame):
        predicted_classes = y_prob.columns[np.argmax(y_prob.values, axis=1)].values
        if input_type is np.ndarray:
            return predicted_classes
        predicted_classes_series = pd.Series(predicted_classes, index=y_prob.index)
        return predicted_classes_series



class PairwiseDifferenceClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """ Works on binary and Multi class classification"""
    estimator = None
    use_prior = True
    X_train_: pd.DataFrame
    y_train_: pd.Series
    sample_weight_: pd.Series = None

    def __init__(
            self,
            estimator=None,
    ):
        if estimator is None:
            from sklearn.ensemble import ExtraTreesClassifier
            estimator = ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)
        else:
            PairwiseDifferenceBase.check_estimator(estimator, expecting_classifier=True)
        if isinstance(estimator, type):
            raise TypeError(
                "estimator must be an instance of the class not a class, i.e., use MyEstimator() but not MyEstimator")
        if not sklearn.base.is_classifier(estimator):
            print('WARNING: estimator must be a Sklearn classifier')
        # todo check it is not multilabel problem: multiclass.is_multilabel: Helper function to check if the task is a multi-label classification one.
        # todo user warning if it is a parametric model i.e. LogisticRegression, NaiveBayes, KNN, SVM, GaussianProcess, etc.
        super().__init__()
        self.estimator = estimator
        self.prior = None
        self.use_prior = 'auto'
        self.proba_aggregate_method = 'norm'

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            # X_val: pd.DataFrame | None = None,
            # y_val: pd.Series | None = None, weight_method='OptimizeOnValidation',
            check_input=True):
        # todo add **karg to pass to the inner model
        self.check_input = check_input
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if self.check_input:
            # X, y = sklearn.utils.validation.check_X_y(X, y)  # Check that X and y have correct shape  # this turn it to ndarray # todo put back
            PairwiseDifferenceBase.check_input(X)
            PairwiseDifferenceBase.check_output(y)
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)  # Store the classes seen during fit

        self.X_train_ = X
        self.y_train_ = y
        self.feature_names_in_ = X.columns
        self.nb_classes_ = self.y_train_.nunique()
        self._estimate_prior()
        X_pair, _ = PairwiseDifferenceBase.pair_input(self.X_train_, self.X_train_)
        y_pair_diff = PairwiseDifferenceBase.pair_output_difference(self.y_train_, self.y_train_, self.nb_classes_)
        # todo add assert on y_pair_diff: min<0  , max>0 and dtype float not uint
        self.estimator.fit(X_pair, y_pair_diff)
        #  plot scatter train improvement vs test improvement

        return self

    def predict_similarity_samples(self, X: pd.DataFrame, X_anchors=None, reshape=True) -> pd.DataFrame:
        """ For each input sample, output C probabilities for each N train pair.
        Beware that this function does not apply the weights at this level
        """
        if X_anchors is None:
            X_anchors = self.X_train_
            # but I also need to change the self.y_train_ at the higher call
        check_is_fitted(self)
        if self.check_input:
            PairwiseDifferenceBase.check_input(X)

        X_pair, X_pair_sym = PairwiseDifferenceBase.pair_input(X, X_anchors)
        if hasattr(self.estimator, 'predict_proba'):
            predict_proba = self.estimator.predict_proba
        else:
            def predict_proba(X) -> np.ndarray:
                predictions = self.estimator.predict(X)
                predictions = predictions.astype(int)
                n_samples = len(predictions)
                proba = np.zeros((n_samples, 2), dtype=float)
                proba[range(n_samples), predictions] = 1.
                return proba

        predictions_proba_difference: np.ndarray = predict_proba(X_pair)
        predictions_proba_difference_sym: np.ndarray = predict_proba(X_pair_sym)
        assert isinstance(predictions_proba_difference, np.ndarray), type(predictions_proba_difference)
        np.testing.assert_array_equal(predictions_proba_difference.shape, (len(X_pair), 2))
        predictions_proba_similarity_ab = predictions_proba_difference[:, 0]
        predictions_proba_similarity_ba = predictions_proba_difference_sym[:, 0]
        predictions_proba_similarity = (predictions_proba_similarity_ab + predictions_proba_similarity_ba) / 2.
        if not reshape:
            return predictions_proba_similarity
        else:
            predictions_proba_similarity_df = pd.DataFrame(
                predictions_proba_similarity.reshape((-1, len(self.X_train_))),
                index=X.index, columns=self.X_train_.index)
            return predictions_proba_similarity_df

    def predict_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('todo use predict_proba_samples to get the prediction here')

    def _estimate_prior(self):
        if self.prior is not None:
            return self
        # Calculate class priors
        class_counts = self.y_train_.value_counts()
        class_priors = class_counts / len(self.y_train_)

        # Convert class priors to a dictionary
        self.prior = class_priors.sort_index().values

    def decide_use_prior(self) -> bool:
        if isinstance(self.use_prior, bool):
            return self.use_prior
        else:
            return not (self.y_train_.value_counts().min() >= 5)

    def predict_proba_samples(self, X: pd.DataFrame) -> np.ndarray:
        # todo add unit test with weight ==[1 1 1 ] and weights = None
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        predictions_proba_similarity_df: pd.DataFrame = self.predict_similarity_samples(X)

        def g(anchor_class: np.ndarray, predicted_similarity: np.ndarray) -> np.ndarray:
            """

            :param anchor_class: array int
            :param predicted_similarity: array float
            :return:
            """
            # likelyhood_per_anchor = (1 - predicted_similarity) * self.prior / (1 - self.prior[anchor_class])  # todo beware of division by zero, output an error directly at the fit step
            # todo beware of division by zero, output an error directly at the fit step
            likelyhood_per_anchor = ((1 - predicted_similarity) / (1 - self.prior[anchor_class]))[:,
                                                                                                  np.newaxis] * self.prior
            likelyhood_per_anchor[np.arange(len(likelyhood_per_anchor)), anchor_class] = predicted_similarity
            return likelyhood_per_anchor

        anchor_class = self.y_train_.astype(int).values

        def f(predictions_proba_similarity: np.ndarray) -> np.ndarray:
            """ Here we focus on one test point.
            Given its similarity probabilities.
            Return the probability for each class"""
            test_i_trains_classes = g(anchor_class=anchor_class, predicted_similarity=predictions_proba_similarity)
            np.testing.assert_array_equal(test_i_trains_classes.shape, (len(self.y_train_), self.nb_classes_))
            # test_i_trains_classes is part of the 3d array that I will not return for now
            np.testing.assert_array_almost_equal(test_i_trains_classes.sum(axis=1), 1.)
            return test_i_trains_classes

        tests_trains_classes_likelihood = np.apply_along_axis(
            f, axis=1, arr=predictions_proba_similarity_df.values)  # todo  guess this is slow, get rid of it

        np.testing.assert_array_equal(tests_trains_classes_likelihood.shape,
                                      (len(X), len(self.y_train_), self.nb_classes_))
        np.testing.assert_array_almost_equal(tests_trains_classes_likelihood.sum(axis=-1), 1.)
        return tests_trains_classes_likelihood

    @staticmethod
    def _apply_weights(tests_trains_classes_likelihood: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
        tests_classes_likelihood = (tests_trains_classes_likelihood * sample_weight[np.newaxis, :, np.newaxis]).sum(
            axis=1)
        np.testing.assert_array_almost_equal(tests_classes_likelihood.sum(axis=-1), 1.)
        return tests_classes_likelihood

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        input_type = type(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.sample_weight_ is not None:
            sample_weight = self.sample_weight_.loc[self.y_train_.index].values
        else:
            sample_weight = np.full(len(self.y_train_), 1 / len(self.y_train_))
        if self.decide_use_prior():
            tests_trains_classes_likelihood = self.predict_proba_samples(X)
            tests_classes_likelihood = self._apply_weights(tests_trains_classes_likelihood, sample_weight)

            assert tests_classes_likelihood.max() < 1.00001, tests_classes_likelihood.max()
            assert tests_classes_likelihood.min() > -.00001, tests_classes_likelihood.min()

            eps = np.finfo(tests_classes_likelihood.dtype).eps
            tests_classes_likelihood = tests_classes_likelihood / tests_classes_likelihood.sum(axis=1)[:, np.newaxis]
            assert np.isclose(tests_classes_likelihood.sum(axis=1), 1, rtol=1e-15, atol=5 * eps).all(), tests_classes_likelihood.sum(axis=1)

            tests_classes_likelihood = tests_classes_likelihood.clip(0, 1)
            if input_type is np.ndarray:
                return tests_classes_likelihood
            tests_classes_likelihood_df = pd.DataFrame(tests_classes_likelihood, index=X.index)
            return tests_classes_likelihood_df
        else:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            predictions_proba_similarity_df: pd.DataFrame = self.predict_similarity_samples(X)
            if self.sample_weight_ is not None:
                raise NotImplementedError()
            else:
                def f(predictions_proba_similarity: pd.Series) -> pd.Series:
                    df = pd.DataFrame({'start': self.y_train_.reset_index(drop=True), 'similarity': predictions_proba_similarity})
                    mean = df.groupby('start', observed=False).mean()['similarity']
                    return mean


                tests_classes_likelihood_np = predictions_proba_similarity_df.apply(f, axis='columns')
                # without this normalization it should work for multiclass-multilabel
                if self.proba_aggregate_method == 'norm':
                    tests_classes_likelihood_np = tests_classes_likelihood_np.values / tests_classes_likelihood_np.values.sum(axis=-1)[:, np.newaxis]
                    # asset if any nan in tests_classes_likelihood_np
                    assert not pd.isna(tests_classes_likelihood_np).any().any()
                    np.testing.assert_almost_equal(np.sum(tests_classes_likelihood_np, axis=-1), 1.)
                elif self.proba_aggregate_method == 'softmax':
                    tests_classes_likelihood_np = softmax(tests_classes_likelihood_np, axis=-1)
                    np.testing.assert_almost_equal(np.sum(tests_classes_likelihood_np, axis=-1), 1.)

                if input_type is np.ndarray:
                    return tests_classes_likelihood_np
                return pd.DataFrame(tests_classes_likelihood_np, index=X.index)

    def predict(self, X) -> pd.Series:
        """ For each input sample, output one prediction the most probable class.
        Predict classes for X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        input_type = type(X)
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                print(X)
                print(X.shape)
                raise
        predict_proba = self.predict_proba(X)
        predict_proba.columns = self.classes_
        return PairwiseDifferenceBase.predict(predict_proba, input_type=input_type)


    def score_difference(self, X, y) -> float:
        """ WE RETURN THE MAE score XD """
        if self.check_input:
            PairwiseDifferenceBase.check_output(y)

        y_pair_diff = PairwiseDifferenceBase.pair_output_difference(y, self.y_train_,
                                                                    self.nb_classes_)  # 0 if similar, 1 if diff
        predictions_proba_similarity: pd.DataFrame = self.predict_similarity_samples(X,
                                                                                     reshape=False)  # 0% if different, 100% if similar

        return abs(y_pair_diff - (1 - predictions_proba_similarity)).mean()


class PairwiseDifferenceRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Pairwise Difference Regressor (PDR) is a meta-estimator that estimates the regression task by estimating the difference between data points.
    PDR estimates the regression task by estimating the distance of the given sample to each of the training samples (the anchors).
    PDR is a modified version implemented by Belaid et al. 2024 of the PAirwise Difference REgressor (Padre) by Tynes et al. 2021

    After fitting, you can use the method `learn_anchor_weights` to learn
    weights for the anchors using the given validation data.
    """
    estimator = None
    X_train_: pd.DataFrame
    y_train_: pd.Series
    sample_weight_: pd.Series = None

    def __init__(
            self,
            estimator: sklearn.base.RegressorMixin = None
    ):
        """
        :param estimator: sklearn regressor for estimating the distance (Default: sklearn.ensemble.HistGradientBoostingRegressor)
        """
        if estimator is None:
            from sklearn.ensemble import ExtraTreesRegressor
            estimator = ExtraTreesRegressor()
        else:
            PairwiseDifferenceBase.check_estimator(estimator, expecting_regressor=True)
        super().__init__()
        self.estimator = estimator

        # Save information about the weighting methods as here for better availability
        self._name_to_method_mapping = {
            # Optimization based methods:
            'L2': functools.partial(self._sample_weight_optimize, l2_lambda=0.1), # Recommended method
            'KLD': functools.partial(self._sample_weight_optimize, kld_lambda=0.05),
            'Optimize': self._sample_weight_optimize,
            'L1L2': functools.partial(self._sample_weight_optimize, l1_lambda=0.05, l2_lambda=0.025),
            'L1': functools.partial(self._sample_weight_optimize, l1_lambda=0.1),
            'ExtremeWeightPruning': self._sample_weight_extreme_pruning,
            # Heuristic methods
            'NegativeError': self._sample_weight_negative_error,
            'InverseError': self._sample_weight_inverse_error,
            'OrderedVoting': self._sample_weight_ordered_votes,
            # Other Methods:
            'KMeansClusterCenters': self._sample_weight_by_kmeans_prototypes,
        }

    @staticmethod
    def _to_pandas(*args):
        return (data if data is None or isinstance(data, (pd.DataFrame, pd.Series)) else pd.DataFrame(data) for data in args)

    @staticmethod
    def _pair_data(X1, X2, y1=None, y2=None):
        X1, y1, X2, y2 = PairwiseDifferenceRegressor._to_pandas(X1, y1, X2, y2)
        assert not is_unsigned_integer_dtype(y1), "_pair_data()'s input y1 is an unsigned integer."
        assert not is_unsigned_integer_dtype(y2), "_pair_data()'s input y2 is an unsigned integer."

        X_pair = X1.merge(X2, how="cross")
        x1_pair = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_diff' for column in X1.columns})
        x2_pair = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_diff' for column in X1.columns})
        X_pair = pd.concat([X_pair, x1_pair - x2_pair], axis='columns')
        # Symmetric
        x2_pair_sym = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_y' for column in X1.columns})
        x1_pair_sym = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_x' for column in X1.columns})
        X_pair_sym = pd.concat([x1_pair_sym, x2_pair_sym, x2_pair - x1_pair], axis='columns')

        if y1 is not None:
            assert isinstance(y1, pd.Series) or y1.shape[1] == 1, f"Didn't expect more than one output {y1.shape}"
            assert isinstance(y2, pd.Series) or y2.shape[1] == 1, f"Didn't expect more than one output {y2.shape}"

            y_pair = pd.DataFrame(y1).merge(y2, how="cross")
            y_pair_diff = y_pair.iloc[:, 0] - y_pair.iloc[:, 1]
        else:
            y_pair_diff = None

        return X_pair, X_pair_sym, y_pair_diff

    @staticmethod
    def _get_pair_feature_names(features: list) -> list:
        """ Get the new name of features after pairing points. """
        return [f'{name}_x' for name in features] + [f'{name}_y' for name in features]

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None, check_input=True):
        """ Transform the data into the pair+difference format and train a ML model. """
        self.check_input = check_input
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if y.name is None:
            # just put any name to the output to avoid a bug later
            y.name = 'output'
        if self.check_input:
            PairwiseDifferenceBase.check_input(X)

        self.X_train_ = X
        self.y_train_ = y
        self.feature_names_in_ = X.columns

        X_pair, _, y_pair_diff = self._pair_data(self.X_train_, self.X_train_, self.y_train_, self.y_train_)
        assert y_pair_diff.abs().max() <= 2 * \
               self.y_train_.abs().max(), f'should be a-(-a) <= 2*a Expected:\n{y_pair_diff.abs().max()} <=  {2 * self.y_train_.abs().max()}'
        # todo add assert on y_pair_diff: min<0  , max>0 and dtype float not uint
        self.estimator.fit(X_pair, y_pair_diff)
        return self

    def _predict_samples(self, X: pd.DataFrame, force_symmetry=True):  # -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each input sample, output N predictions (where N = the number of anchors).
        prediction = difference + y_train
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        assert len(X.shape) == 2

        # Create pairs of the new instance each anchor (training instance)
        X_pair, X_pair_sym, _ = self._pair_data(X, self.X_train_, None, None)
        assert isinstance(self.y_train_, (pd.Series, pd.DataFrame)), type(self.y_train_)

        def repeat(s: pd.Series, n_times: int):
            return pd.concat([s] * n_times, ignore_index=True).values

        # Estimator predicts the difference between each anchor (training instance) and each prediction instance:
        predictions_difference: np.ndarray = self.estimator.predict(X_pair)
        assert isinstance(predictions_difference, np.ndarray), type(predictions_difference)
        if force_symmetry:
            difference_sym: np.ndarray = self.estimator.predict(X_pair_sym)
            predictions_difference = (predictions_difference - difference_sym) / 2.

        # The known y for the training instances
        predictions_start: np.ndarray = repeat(self.y_train_, n_times=len(X))
        assert isinstance(predictions_start, np.ndarray), type(predictions_start)

        # Combine the difference predicted by the model with the known y => train_y + predicted difference
        predictions: np.ndarray = predictions_start + predictions_difference
        assert isinstance(predictions, np.ndarray), type(predictions)

        # Set of absolute predictions for each anchor for each prediction instance:
        prediction_samples_df = pd.DataFrame(predictions.reshape((-1, len(self.X_train_))), index=X.index,
                                             columns=self.X_train_.index)
        # The predicted difference to the anchors:
        pred_diff_samples_df = pd.DataFrame(predictions_difference.reshape((-1, len(self.X_train_))), index=X.index,
                                            columns=self.X_train_.index)
        return prediction_samples_df, pred_diff_samples_df

    # def _predict_distribution(self, X) -> pd.DataFrame:  # maybe outdated that's why I comment it until I check it
    #     """ For each input sample, output one pair (mean, std) of the predicted samples.
    #     The return is of shape pd.Series"""
    #     prediction_samples_df = self._predict_samples(X)
    #     prediction_stats = prediction_samples_df.apply(lambda samples: (samples.mean(), samples.std(df=1)), axis='columns')
    #     prediction_stats.columns = ['mean', 'std']
    #     return prediction_stats

    def predict(self, X, force_symmetry=True) -> pd.Series:
        """ For each input sample, output one prediction, the mean of the predicted samples. """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        prediction_samples_df, _ = self._predict_samples(X, force_symmetry=force_symmetry)

        if self.sample_weight_ is None: # No weights
            prediction_stats = prediction_samples_df.mean(axis=1)

        elif isinstance(self.sample_weight_, pd.Series):
            def weighted_avg(samples: pd.Series, weights: pd.Series) -> float:
                weights[weights <= 0] = np.nan
                summed = np.nansum(samples.multiply(weights))
                return  summed / np.nansum(weights)

            prediction_stats = prediction_samples_df.apply(
                lambda samples: weighted_avg(samples, self.sample_weight_),
                axis='columns'
            )

        elif isinstance(self.sample_weight_, pd.DataFrame):
            self.sample_weight_[self.sample_weight_ < 0] = np.nan
            summed = pd.Series(np.nansum(self.sample_weight_, axis=1), index=X.index)
            self.sample_weight_ = self.sample_weight_.apply(lambda row: row / summed)
            np.testing.assert_array_almost_equal(self.sample_weight_.sum(axis=1), 1.)
            prediction_stats = (prediction_samples_df * self.sample_weight_).sum(axis=1)

        else:
            raise NotImplementedError("weights must be a pd.Series or a pd.DataFrame")

        assert len(X) == len(prediction_stats)
        assert not any(pd.isna(prediction_stats)), f'Prediction should not contain Nans\n{prediction_samples_df}\n{prediction_stats}'
        prediction_stats = pd.Series(prediction_stats, index=X.index)
        return prediction_stats

    def learn_anchor_weights(
            self,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            X_test: pd.DataFrame = None,
            method: str = 'L2',
            enable_warnings=True,
            **kwargs):
        """
        Call this method after the training to create weights for the anchors
        using the given validation data.
        Use the `method` parameter to select one of the following
        weighting methods:
        - 'Optimize': Minimize the validation MAE using the SLSQP optimizer with a linear constraint on the sum of the weights.
        - 'L1': like `Optimize` but includes L1 regularization.
        - 'L2': like `Optimize` but includes L2 regularization.
        - 'L1L2': like `Optimize` but includes L1 and L2 regularization.
        - 'KLD': like `Optimize` but includes a KLD loss to make the weights more uniform.
        - 'ExtremeWeightPruning': lik `L1` but uses  high regularization strength.
        - 'NegativeError': Calculate weights as the negative mean absolute error.
        - 'OrderedVoting': The best of n anchors gets n votes, the worst gets 1 vote. n is the number of anchors.
        - 'KmeansClusterCenters': Calculate weights as the distance to the cluster centers of the KMeans algorithm.

        :param X_val: X of a validation set
        :param y_val: y of a validation set
        :param X_test: y of the test set
        :param method: one of
        :param enable_warnings: Set to true if you want to be warned about changing scores (Default: True)
        :param kwargs: Additional parameters for the weighting method (e.g. K for k-nearest neighbors)
        :return: self (with updated weights)
        """
        # todo merge into fit like sklearn
        if y_val is not None:
            old_validation_error = sklearn.metrics.mean_absolute_error(self.predict(X_val), y_val)
        else:
            old_validation_error = 0

        if method not in self._name_to_method_mapping.keys():
            raise NotImplementedError(f"Weighting method {method} unknown! Use one of the following:"
                                      f" '{', '.join(list(self._name_to_method_mapping.keys()))}'")

        sample_weight: pd.Series = self._name_to_method_mapping[method](X_val=X_val, y_val=y_val, X_test=X_test, **kwargs)
        assert not sample_weight.isna().any(), f'Nans values in sample_weights using {method}\n {sample_weight}'
        self.set_sample_weight(sample_weight)
        if y_val is not None:
            new_validation_error = sklearn.metrics.mean_absolute_error(self.predict(X_val), y_val)
            if new_validation_error > old_validation_error and enable_warnings:
                print(f'WARNING: \t new val MAE: {new_validation_error} \t old val MAE:  {old_validation_error}')
        return self

    def set_sample_weight(self, sample_weight: pd.Series):
        """
        Sets the weights for the anchors to the given weights in sample_weight.

        :param sample_weight: The weights for the anchors as a pd.Series
        :return: self (with updated weights)
        """
        if sample_weight is None:
            pass
        elif isinstance(sample_weight, pd.Series):
            if len(sample_weight) != len(self.y_train_):
                raise ValueError(
                    f'sample_weight size {len(sample_weight)} should be equal to the train size {len(self.y_train_)}')
            if not sample_weight.index.equals(self.y_train_.index):
                raise ValueError(
                    f'sample_weight and y_train must have the same index\n{sample_weight.index}\n{self.y_train_.index}')

            if all(sample_weight.fillna(0) == 0):  # All weights are 0 => Set them to 1
                sample_weight = pd.Series(1, index=self.y_train_.index)

            if all(sample_weight.fillna(0) < 0):
                raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')
            if any(pd.isna(sample_weight)):
                raise ValueError(f'sample_weight contains NaNs.\n{sample_weight}')
        else:
            raise ValueError('sample_weight must be a pd.Series')

        self.sample_weight_ = sample_weight
        return self

    @staticmethod
    def _normalize_weights_to_0_to_1(weights: pd.Series) -> pd.Series:
        """
        Normalize the weights to be between 0 and 1
        :param weights: The weights to be normalized as a pd.Series
        """
        if all(np.isclose(weights, weights.values[0])):
            weights = pd.Series(1., index=weights.index)
        assert weights.min() >= 0, f'Negative weights found: {weights[weights < 0]}'
        weights /= weights.sum()
        return weights

    @staticmethod
    def __objective_function(weights: np.ndarray, pred_val_samples_np: np.ndarray, y_val: np.ndarray, initial_mae: float, kld_lambda=0., l1_lambda=0., l2_lambda=0.) -> float:
        assert kld_lambda >= 0, f'kld_lambda should be >=0, got {kld_lambda}'
        assert l1_lambda >= 0, f'l1_lambda should be >=0, got {l1_lambda}'
        assert l2_lambda >= 0, f'l2_lambda should be >=0, got {l2_lambda}'
        assert initial_mae >= 0, f'initial_mae should be >=0, got {initial_mae}'

        predictions = np.matmul(pred_val_samples_np, weights / sum(weights))
        mae = sklearn.metrics.mean_absolute_error(y_val, predictions)

        regularisation = 0
        if kld_lambda > 0:
            train_size = len(weights)
            weights_initial_guess = np.ones(train_size) / train_size
            regularisation += kld_lambda * entropy(weights, weights_initial_guess) / train_size
        if l1_lambda > 0:
            regularisation += l1_lambda * (np.linalg.norm(weights, ord=1) - max(weights))
        if l2_lambda > 0:
            regularisation += l2_lambda * np.linalg.norm(weights, ord=2)

        regularisation *= initial_mae
        loss = mae + regularisation
        return loss

    def _sample_weight_optimize(self, X_val: pd.DataFrame, y_val: pd.Series, kld_lambda=0., l1_lambda=0., l2_lambda=0., **kwargs) -> pd.Series:
        """
        Minimize the validation MAE using SLSQP optimizer
        with a linear constraint on the sum of the weights.

        :param X_val:
        :param y_val:
        :param kld_lambda: alpha=0.01 i.e. I am ready to loose 1% of the validation MAE to make the solution more general
        :return:
        """
        prediction_samples_df, _ = self._predict_samples(X_val)
        pred_val_samples_np = prediction_samples_df.values
        train_size = len(self.X_train_)
        weights_initial_guess = np.ones(train_size) / train_size
        initial_mae = sklearn.metrics.mean_absolute_error(y_val, np.matmul(pred_val_samples_np, weights_initial_guess))

        def objective_function(weights: np.ndarray) -> float:
            return self.__objective_function(weights=weights, pred_val_samples_np=pred_val_samples_np, y_val=y_val, initial_mae=initial_mae, kld_lambda=kld_lambda, l1_lambda=l1_lambda, l2_lambda=l2_lambda)

        variable_bounds = [(0., 1.) for _ in range(train_size)]
        sum_constraint = LinearConstraint(np.ones(train_size), lb=1, ub=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(objective_function, weights_initial_guess, method='SLSQP', bounds=variable_bounds, constraints=[sum_constraint])
        # Extract the solution
        optimal_weight = result.x

        # print("the optimal solution:", optimal_weight)
        # print("Optimal Objective Value, i.e. new log loss validation error:", result.fun)
        sample_weights = pd.Series(optimal_weight, index=self.X_train_.index)
        return sample_weights

    def _sample_weight_extreme_pruning(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        l1 = 0.8
        while l1 > 0.0001:
            weights = self._sample_weight_optimize(X_val=X_val, y_val=y_val, l1_lambda=l1)
            if sum(weights == 0) / len(weights) > .9:
                l1 *= 0.5
            else:
                break
        return weights

    def _error(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate the Mean Absolute Error for each anchor.
        :param X_val:
        :param y_val:
        :param kwargs:
        :return:
        """
        pred_val_samples, _ = self._predict_samples(X_val)
        errors = pred_val_samples.apply(lambda one_val_samples: abs(y_val - one_val_samples), axis=0)
        val_mae = errors.mean()
        np.testing.assert_array_equal(val_mae.index, self.X_train_.index)
        return val_mae

    def _sample_weight_inverse_error(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        val_mae = self._error(X_val=X_val, y_val=y_val)
        sample_weights = 1. / (val_mae + 0.0001)
        sample_weights = sample_weights / sample_weights.sum()
        return sample_weights

    def _sample_weight_negative_error(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        uniform_weights = pd.Series([1 / len(self.X_train_)] * len(self.X_train_), index=self.X_train_.index)
        val_mae = self._error(X_val=X_val, y_val=y_val)
        if sum(val_mae) == 0:
            return uniform_weights
        sample_weights = ((-val_mae) + max(val_mae)) / sum(val_mae)
        if sum(sample_weights) == 0:
            return uniform_weights
        sample_weights = sample_weights / sample_weights.sum()
        return sample_weights

    @staticmethod
    def _sample_weight_ordered_votes_from_weights(received_weights):
        errors = - received_weights
        k = len(errors)
        ranks = np.argsort(np.argsort(errors)) + 1
        weights = (k - ranks + 1) / (k * (k + 1) / 2)
        return weights

    def _sample_weight_ordered_votes(self, X_val, y_val, force_symmetry=True, **kwargs):
        """
        The best of n anchors gets n votes, the worst gets 1 vote. n is the nb of anchors. Uses the _sample_weight_negative_error function
        for distribution votes.
        works quite good
        :param force_symmetry: Sets the force_symmetry parameter of the prediction function
        :return: The weights as np.NDarray
        """
        weights = self._sample_weight_negative_error(X_val, y_val, force_symmetry=force_symmetry)
        return self._sample_weight_ordered_votes_from_weights(weights)

    def _sample_weight_by_kmeans_prototypes(self, k=None, **kwargs):
        """
        Use KMeans to cluster the train data. Use the k centroids/prototypes found by knn as weights.
        We keep only K anchors that are the prototypes. all other anchors receive a weight of 0

        :param force_symmetry: Sets the force_symmetry parameter of the prediction function
        :param k: The number of prototypes to use. If None, 10% of the training set is used as prototypes
        :return: The weights as np.NDarray
        """
        if not k:
            k = max(int(len(self.X_train_) / 10), 3)  # 10% and min 3 of the training set data points is used as weights

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        kmeans.fit(self.X_train_)

        cluster_centers = kmeans.cluster_centers_  # Get the cluster centers (prototypical data points)
        distances = cdist(self.X_train_, cluster_centers)  # distance between each data point and each cluster center
        closest_indices = np.argmin(distances, axis=0)  # Get the index of the closest data points to the clusters

        # Create an array to mark the closest data points
        closest_array = np.zeros(len(self.X_train_))
        closest_array[closest_indices] = 1 / k

        s = pd.Series(closest_array, index=self.X_train_.index)
        s = s.fillna(0)  # I don't know why there are NaNs rather than 0s
        assert not s.isna().any(), f'Nans values in sample_weights using KMeans\n {s}'
        return s

class PDCDataTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Transform the data so that it can be processed by PDL models.
    """
    preprocessing_: ColumnTransformer
    preprocessing_y_: ColumnTransformer # todo fix the ColumnTransformer annotation

    def __init__(self, numeric_features:Iterable=None, ordinal_features:Iterable=None, string_features:Iterable=None, y_type:str=None):
        self.numeric_features = numeric_features
        self.ordinal_features = ordinal_features
        self.string_features = string_features
        if y_type is not None and y_type not in ('numeric', 'ordinal', 'string'):
            raise ValueError(f"y_type must be one of 'numeric', 'ordinal', 'string' but got {y_type}")
        self.y_type = y_type

    def fit(self, X, y=None):

        # X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

        # Data processing

        # y = y.astype('category').cat.codes.astype(np.float32) # todo since I cannot transform the output at least add raise type error on it
        if self.numeric_features is None and self.ordinal_features is None and self.string_features is None:
            self.numeric_features = []
            self.ordinal_features = []  # todo fix name, will be processed a ordinal
            self.string_features = []
            for column in X.columns:
                dtype = X[column].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    self.numeric_features.append(column)
                elif isinstance(dtype, pd.CategoricalDtype):
                    if dtype.ordered:
                        self.ordinal_features.append(column)  # ordinal...
                    else:
                        self.string_features.append(column)
                elif pd.api.types.is_bool_dtype(dtype):  # pd.api.types.is_categorical_dtype(dtype) deprecated
                        self.string_features.append(column)
                elif pd.api.types.is_string_dtype(dtype):
                    self.string_features.append(column)

        from benchmark.benchmark_utils import cast_uint, get_generic_preprocessing
        X, _ = cast_uint(X)
        self.preprocessing_ = get_generic_preprocessing(self.numeric_features, self.ordinal_features, self.string_features)
        self.preprocessing_.fit(X)

        self.preprocessing_y_ = None
        if self.y_type == 'numeric':
            from sklearn.preprocessing import StandardScaler
            self.preprocessing_y_ = StandardScaler()
        elif self.y_type == 'ordinal': #  string
            from sklearn.preprocessing import OrdinalEncoder
            self.preprocessing_y_ = OrdinalEncoder()
        elif self.y_type == 'string':
            from sklearn.preprocessing import OneHotEncoder
            self.preprocessing_y_ = OneHotEncoder()

        if y is not None and self.preprocessing_y_ is not None:
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            self.preprocessing_y_.fit(y)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        from benchmark.benchmark_utils import cast_uint
        X, _ = cast_uint(X)
        X = pd.DataFrame(self.preprocessing_.transform(X))
        from scipy.sparse import csr_matrix
        if any(isinstance(e, csr_matrix) for e in X.values.flatten()):
            raise NotImplementedError('error in data \t X contains sparse features (csr_matrix)')
        X = X.dropna(axis=1, how='all')  # Drop columns with all NaN values
        X = X.astype(np.float32)

        if len(X.columns) == 0:
            raise ValueError('error in data \t X no features left after pre-processing')
        # if X.isna().any().any():
        #     raise NotImplementedError('error in data \t Some features are NaNs in the X set')
        if any(x in pd.Series(X.values.flatten()).apply(type).unique() for x in ('csr_matrix', 'date',)):  # todo think about adding  'str'
            raise NotImplementedError('error in data \t Dataset contains sparse data')

        if y is not None and self.preprocessing_ is not None:
            y = pd.Series(self.preprocessing_.transform(y), name='y')
        if y is None:
            return X.values
        return X.values, y.values
