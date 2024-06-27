"""Pairwise Difference Learning meta-estimator."""

# Author: Mohamed Karim Belaid <karim.belaid@idiada.com> or <extern.karim.belaid@porsche.de>
# License: Apache-2.0 clause

import numpy as np
import pandas as pd
import sklearn.base
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
        X_pair = pd.concat([X_pair, x1_pair - x2_pair], axis='columns')
        # Symmetric
        x2_pair_sym = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_y' for column in X1.columns})
        x1_pair_sym = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_x' for column in X1.columns})
        X_pair_sym = pd.concat([x1_pair_sym, x2_pair_sym, x2_pair - x1_pair], axis='columns')

        return X_pair, X_pair_sym

    @staticmethod
    def pair_output(y1: pd.Series, y2: pd.Series) -> pd.Series:
        """For regression. Beware this is different from regression this is b-a not a-b"""
        y_pair = pd.DataFrame(y1).merge(y2, how="cross")
        y_pair_diff = y_pair.iloc[:, 1] - y_pair.iloc[:, 0]
        return y_pair_diff

    @staticmethod
    def pair_output_difference(y1: pd.Series, y2: pd.Series, nb_classes: int) -> pd.Series:
        """For MultiClassClassification base on difference only"""
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
    def predict(y_prob, input_type=pd.DataFrame):
        predicted_classes = np.argmax(y_prob.values, axis=1)
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

    def __init__(
            self,
            estimator=None,
    ):
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier()
        if isinstance(estimator, type):
            raise TypeError("estimator must be an instance of the class not a class, i.e., use MyEstimator() but not MyEstimator")
        if not sklearn.base.is_classifier(estimator):
            print('WARNING: estimator must be a Sklearn classifier')
        # todo check it is not multilabel problem: multiclass.is_multilabel: Helper function to check if the task is a multi-label classification one.
        # todo user warning if it is a parametric model i.e. LogisticRegression, NaiveBayes, KNN, SVM, GaussianProcess, etc.
        super().__init__()
        self.estimator = estimator
        self.prior = None
        self.use_prior = 'auto'

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
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

    def decide_use_prior(self) -> bool:
        if isinstance(self.use_prior, bool):
            return self.use_prior
        else:
            return not (self.y_train_.value_counts().min() >= 5)

    def predict_proba_samples(self, X: pd.DataFrame) -> np.ndarray:
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

            def f(predictions_proba_similarity: pd.Series) -> pd.Series:
                df = pd.DataFrame({'start': self.y_train_, 'similarity': predictions_proba_similarity})
                mean = df.groupby('start').mean()['similarity']
                return mean

            tests_classes_likelihood_np = predictions_proba_similarity_df.apply(f, axis='columns')
            tests_classes_likelihood_np = softmax(tests_classes_likelihood_np, axis=-1)
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


class PairwiseDifferenceRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Pairwise Difference Regressor (PDR) is a meta-estimator that estimates the regression task by estimating the difference between data points.
    PDR estimates the regression task by estimating the distance of the given sample to each of the training samples (the anchors).
    PDR is a modified version implemented by Belaid et al. 2024 of the PAirwise Difference REgressor (Padre) by Tynes et al. 2021
    """
    estimator = None
    X_train_: pd.DataFrame
    y_train_: pd.Series

    def __init__(
            self,
            estimator: sklearn.base.RegressorMixin
    ):
        """
        :param estimator: sklearn regressor for estimating the distance (Default: sklearn.ensemble.HistGradientBoostingRegressor)
        """
        super().__init__()
        self.estimator = estimator

        if estimator is None:  # Set default
            self.estimator = sklearn.ensemble.HistGradientBoostingRegressor()

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if y.name is None:
            # just put any name to the output to avoid a bug later
            y.name = 'output'

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

        prediction_stats = prediction_samples_df.mean(axis=1)

        assert len(X) == len(prediction_stats)
        assert not any(pd.isna(prediction_stats)), f'Prediction should not contain Nans\n{prediction_samples_df}\n{prediction_stats}'
        prediction_stats = pd.Series(prediction_stats, index=X.index)
        return prediction_stats
