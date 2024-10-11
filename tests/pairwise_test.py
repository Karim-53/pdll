import os
import unittest
import logging
import openml
from openml.extensions import get_extension_by_model
import sys
# print(os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./'))  # make sure to use the local version of pdll
from pdll._pairwise import PairwiseDifferenceClassifier, PDCDataTransformer, PairwiseDifferenceRegressor
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from benchmark.benchmark_utils import group_features

openml.config.set_console_log_level(logging.INFO)

class PDLTest(unittest.TestCase):

    def test_pdl_classifier_on_openml(self):
        # OpenML test server
        openml.config.start_using_configuration_for_example()
        task = openml.tasks.get_task(1196)
        # todo assert nb of clases >=2
        dataset = openml.datasets.get_dataset(task.dataset_id, download_qualities=True, download_features_meta_data=False, download_data=True)
        assert dataset.qualities['NumberOfInstances'] <= 11000, f'error in data ID: {task.dataset_id} \t Dataset is too large and would slow the experiment'

        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        # fill X['sepallength'] with strings
        X['sepallength'] = X['sepallength'].astype(str)
        print('input data shape', X.shape)
        numeric_features, nominal_features, string_features = group_features(dataset.features, X.columns)
        data_transformer = PDCDataTransformer(numeric_features, nominal_features, string_features)
        clf = PairwiseDifferenceClassifier(estimator=DecisionTreeClassifier(class_weight="balanced"))
        clf = Pipeline(steps=[
            ('pre-processor', data_transformer),
            ('PDL.Classifier', clf),
        ])
        extension = get_extension_by_model(clf, raise_if_no_extension=True)
        flow = extension.model_to_flow(clf)
        # flow_id = flow.publish()
        # clf.fit(X,y)
        # print(clf.predict(X))
        print('run_model_on_task...')
        run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=True)
        print('run_model_on_task...done')
        print(run)
        assert float(run._get_repr_body_fields()[2][1][:6]) > .90, "Obtained accuracy too low, expected 0.9533 +- 0.0521"

    def test_pdl_regressor_on_openml(self):
        # OpenML test server
        openml.config.start_using_configuration_for_example()
        task = openml.tasks.get_task(619)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_qualities=True, download_features_meta_data=False, download_data=True)
        assert dataset.qualities['NumberOfInstances'] <= 200, f'error in data ID: {task.dataset_id} \t Dataset is too large and would slow the experiment'

        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        print('input data shape', X.shape)
        numeric_features, nominal_features, string_features = group_features(dataset.features, X.columns)
        data_transformer = PDCDataTransformer(numeric_features, nominal_features, string_features)
        reg = PairwiseDifferenceRegressor(estimator=DecisionTreeRegressor())
        clf = Pipeline(steps=[
            ('pre-processor', data_transformer),
            ('PDL.Regressor', reg),
        ])
        extension = get_extension_by_model(clf, raise_if_no_extension=True)
        flow = extension.model_to_flow(clf)
        # flow_id = flow.publish()
        # clf.fit(X,y)
        # print(clf.predict(X))
        print('run_model_on_task...')
        run = openml.runs.run_model_on_task(clf, task, avoid_duplicate_runs=True)
        print('run_model_on_task...done')
        print(run)
        assert float(run._get_repr_body_fields()[2][1][:6]) < .40, "Obtained MAE is too high, expected 0.2813 +- 0.0873"
