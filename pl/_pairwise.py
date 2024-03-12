"""Pairwise Difference Learning meta-estimator."""

# Author: Mohamed Karim Belaid <karim.belaid@idiada.com> or <extern.karim.belaid@porsche.de>
# License: CC BY-ND 4.0 clause

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.utils.validation import check_is_fitted

# todo Developing scikit-learn estimators: https://scikit-learn.org/stable/developers/develop.html    and this for common term    https://scikit-learn.org/stable/glossary.html
# todo follow this https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html#sphx-glr-auto-examples-developing-estimators-sklearn-is-fitted-py
# todo PairwiseDifference class that can detect if it is classifiaction or regression if it implement predict proba

class PairwiseDifferenceBase(sklearn.base.BaseEstimator):
    @staticmethod
    def pair_input(X1, X2) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_pair = X1.merge(X2, how="cross")
        x1_pair = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_diff' for column in X1.columns})
        x2_pair = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_diff' for column in X1.columns})
        X_pair = pd.concat([X_pair, x1_pair - x2_pair], axis='columns')
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
        # todo check if this is not https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html#sklearn.metrics.pairwise.paired_distances
        y_pair_diff = PairwiseDifferenceBase.pair_output(y1, y2)
        y_pair_diff[y_pair_diff != 0] = 1
        assert y_pair_diff.nunique() <= 2, f'should only be 0 and 1 {y_pair_diff.unique()}'
        return y_pair_diff


    @staticmethod
    def get_pair_feature_names(features: list) -> list:
        """ Get the new name of features after pairing points. """
        return [f'{name}_x' for name in features] + [f'{name}_y' for name in features]

    @staticmethod
    def check_input(X: pd.DataFrame) -> None:
        # todo use https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y
        assert X is not None
        assert isinstance(X, pd.DataFrame)
        assert 'uint' not in str(X.dtypes), X.dtypes
        # todo write more informative error msg
        # todo turn assert into raise val err
        assert len(X.shape) == 2

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
    def check_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> None:
        if sample_weight is None:
            pass
        elif isinstance(sample_weight, pd.Series):
            # check
            if len(sample_weight) != len(y_train):
                raise ValueError(f'sample_weight size {len(sample_weight)} should be equal to the train size {len(y_train)}')
            if not sample_weight.index.equals(y_train.index):
                raise ValueError(f'sample_weight and y_train must have the same index\n{sample_weight.index}\n{y_train.index}')
            if all(sample_weight.fillna(0) <= 0):
                raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')

            # norm
            class_sums = np.bincount(y_train, sample_weight)
            sample_weight = sample_weight / class_sums[y_train.astype(np.int)]
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
        predicted_classes = np.argmax(y_prob.values, axis=1)
        if input_type is np.ndarray:
            return predicted_classes
        predicted_classes_series = pd.Series(predicted_classes, index=y_prob.index)
        return predicted_classes_series


class PairwiseDifferenceClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """ Works on binary and Multi class classification"""
    estimator = None
    X_train_: pd.DataFrame
    y_train_: pd.Series
    sample_weight_: pd.Series

    def __init__(
        self,
        estimator,
    ):
        if isinstance(estimator, type):
            raise TypeError("estimator must be an instance of the class not a class, i.e., use MyEstimator() but not MyEstimator")
        if not sklearn.base.is_classifier(estimator):
            print('WARNING: estimator must be a Sklearn classifier')
        # todo check it is not multilabel problem: multiclass.is_multilabel: Helper function to check if the task is a multi-label classification one.
        # todo user warning if it is a parametric model i.e. LogisticRegression, NaiveBayes, KNN, SVM, GaussianProcess, etc.
        super().__init__()
        self.estimator = estimator
        self.prior = None

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None, weight_method='OptimizeOnValidation',
            check_input=True):
        # todo add the possibility to change the anchor set
        #  todo change the sample weights to anchor weights
        # todo add verbose param to print train size, inner train score, pairwise's train score, before and after, val scores before and after
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
        self.sample_weight_ = None
        return self

    def set_sample_weight(self, sample_weight):
        PairwiseDifferenceBase.check_sample_weight(sample_weight, self.y_train_)
        sample_weight = PairwiseDifferenceBase.correct_sample_weight(sample_weight, self.y_train_)
        PairwiseDifferenceBase.check_sample_weight(sample_weight, self.y_train_)
        self.sample_weight_ = sample_weight
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
        # todo implement the case where the base classifier do not have .predict_proba()
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
            predictions_proba_similarity_df = pd.DataFrame(predictions_proba_similarity.reshape((-1, len(self.X_train_))),
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
            likelyhood_per_anchor = ((1 - predicted_similarity) / (1 - self.prior[anchor_class]))[:, np.newaxis] * self.prior
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
            # test_i_classes = np.sum(test_i_trains_classes*sample_weight[:, np.newaxis]).mean(axis=0)
            # np.testing.assert_almost_equal(sum(test_i_classes), 1.)
            # return test_i_classes
            # return pd.Series(test_i_classes)

        tests_trains_classes_likelihood = np.apply_along_axis(
            f, axis=1, arr=predictions_proba_similarity_df.values)  # todo  guess this is slow, get rid of it

        np.testing.assert_array_equal(tests_trains_classes_likelihood.shape, (len(X), len(self.y_train_), self.nb_classes_))
        np.testing.assert_array_almost_equal(tests_trains_classes_likelihood.sum(axis=-1), 1.)
        return tests_trains_classes_likelihood

    @staticmethod
    def _apply_weights(tests_trains_classes_likelihood: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
        tests_classes_likelihood = (tests_trains_classes_likelihood * sample_weight[np.newaxis, :, np.newaxis]).sum(axis=1)
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
            X = pd.DataFrame(X)
        predict_proba = self.predict_proba(X)
        return PairwiseDifferenceBase.predict(predict_proba, input_type=input_type)

    def score_difference(self, X, y) -> float:
        """ WE RETURN THE MAE score XD """
        if self.check_input:
            PairwiseDifferenceBase.check_output(y)

        y_pair_diff = PairwiseDifferenceBase.pair_output_difference(y, self.y_train_, self.nb_classes_)  # 0 if similar, 1 if diff
        predictions_proba_similarity: pd.DataFrame = self.predict_similarity_samples(X, reshape=False)  # 0% if different, 100% if similar

        return abs(y_pair_diff - (1 - predictions_proba_similarity)).mean()
