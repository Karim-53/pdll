"""Predictions must not depend on the pandas index of the training data.

Regression tests for the bug in PairwiseDifferenceClassifier.predict_proba (pdll <= 0.3.1):
anchor labels were re-indexed 0..N-1 while the similarity row kept the original X_train index,
so pandas paired them by label instead of by position. Any X/y straight out of
train_test_split (shuffled index) gave near-random predictions.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # use the local pdll
from pdll import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor


def wine_split():
    X, y = load_wine(return_X_y=True, as_frame=True)  # 178 rows, 3 classes (>= 5 rows each -> default path)
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def make_pdc():
    return PairwiseDifferenceClassifier(estimator=ExtraTreesClassifier(n_estimators=20, class_weight='balanced', random_state=0))


def proba(X_train, y_train, X_test):
    return np.asarray(make_pdc().fit(X_train, y_train).predict_proba(X_test))


class PandasIndexTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te = wine_split()
        cls.reference = proba(cls.X_tr.reset_index(drop=True), cls.y_tr.reset_index(drop=True), cls.X_te)

    def test_shuffled_index_gives_same_proba_as_reset_index(self):
        self.assertFalse(self.y_tr.index.equals(pd.RangeIndex(len(self.y_tr))), 'precondition: index is shuffled')
        np.testing.assert_allclose(proba(self.X_tr, self.y_tr, self.X_te), self.reference)

    def test_numpy_input_gives_same_proba_as_pandas(self):
        np.testing.assert_allclose(proba(self.X_tr.to_numpy(), self.y_tr.to_numpy(), self.X_te.to_numpy()), self.reference)

    def test_string_index(self):
        X_tr = self.X_tr.set_axis([f'row{i}' for i in self.X_tr.index])
        y_tr = self.y_tr.set_axis(X_tr.index)
        np.testing.assert_allclose(proba(X_tr, y_tr, self.X_te), self.reference)

    def test_shuffled_index_accuracy(self):
        y_pred = np.asarray(make_pdc().fit(self.X_tr, self.y_tr).predict(self.X_te))
        self.assertGreaterEqual((y_pred == self.y_te.to_numpy()).mean(), 0.9)

    def test_proba_rows_sum_to_one(self):
        np.testing.assert_allclose(proba(self.X_tr, self.y_tr, self.X_te).sum(axis=1), 1.)

    def test_prior_path_with_shuffled_index(self):
        """A class with < 5 training rows switches predict_proba to the prior-based path."""
        keep = (self.y_tr != 2) | (self.y_tr.groupby(self.y_tr).cumcount() < 3)
        X_tr, y_tr = self.X_tr[keep], self.y_tr[keep]
        self.assertTrue(make_pdc().fit(X_tr, y_tr).decide_use_prior())
        np.testing.assert_allclose(proba(X_tr, y_tr, self.X_te),
                                   proba(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), self.X_te))

    def test_regressor_shuffled_index(self):
        X, y = self.X_tr.drop(columns='alcohol'), self.X_tr['alcohol']
        def predict(X_train, y_train):
            pdr = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor(random_state=0))
            return np.asarray(pdr.fit(X_train, y_train).predict(self.X_te.drop(columns='alcohol')))
        np.testing.assert_allclose(predict(X, y), predict(X.reset_index(drop=True), y.reset_index(drop=True)))


if __name__ == '__main__':
    unittest.main()
