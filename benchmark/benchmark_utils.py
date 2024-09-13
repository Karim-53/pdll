from joblib import effective_n_jobs
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, BaggingClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import sys
# import tpot
import random
import json
import numpy as np
import openml
import psutil
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from benchmark.configuration_space import *

def decode_args():
    import argparse

    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--iter', type=int, help='iteration number')
    parser.add_argument('--model', nargs="+", type=str, help='Model name')
    parser.add_argument('--regressor', type=str, help='Regressor name, default is DecisionTreeRegressor')
    parser.add_argument('--data', nargs="+", type=int, help='Data value')
    parser.add_argument('--n_jobs', type=int, help='modify n_jobs')  # deprecated
    parser.add_argument('--hyper', action='store_true', help='Hyperparameter')
    parser.add_argument('--quicktest', help='quick test', action='store_true')

    # Parse the arguments
    args = parser.parse_known_args()
    # Convert arguments to dictionary
    arguments_dict = vars(args[0])

    return arguments_dict


args = decode_args()
HYPER_PARAM_OPT = True or args['hyper']
DATASET_SIZE_PAPER = 100 if HYPER_PARAM_OPT else 450
INNER_FOLDS = 3
REPEATS = 5
FOLDS = 5
classifier_dict = {
    # good                                                                        # nohyper
    0: 'DecisionTreeClassifier',          # HPO on serial:  1 min to 3 min        # 3h   OK                             13 mahiech paralelisable, 3maltha *10 w najem nzidha*3
    # HPO on serial: 20 min to 6 hours      # to 6h      to 375Gb  memory starting from 92       paralelilisable w 3andou n_jobs=-1 but do not exploit all cpus
    1: 'RandomForestClassifier',
    2: 'ExtraTreeClassifier',             # HPO on serial:  1 min to 2 min        # 1h   memory starting from 88
    3: 'HistGradientBoostingClassifier',  # HPO on serial:  3 h   to 5 hours      # 9h?      cpu1.8%                                paralelilisable without having n_jobs=-1
    4: 'BaggingClassifier',               # HPO on serial:  5 min to 35min        # 20h?  cpu1.8%                       100 w 100M
    5: 'ExtraTreesClassifier',            # HPO on serial: 15 min to 45min        # 2h memory starting from 125
    6: 'GradientBoostingClassifier',      # HPO on serial:  4 h   to 60h?         # 5h?                               12

    # theoretically impossible to get a good score
    7: 'MLPClassifier',                   # HPO on serial: 7min to 12min          # 15h?             50
    8: 'KNeighborsClassifier',          # HPO on serial: 1min to 17min            #  5h?              100 w 400M
    9: 'GaussianNB',                    # HPO on serial: 1min to 1min             #  1h  OK                 16 and only one hyperparam
    10: 'BernoulliNB',                  # HPO on serial: 1min to 1min             #  5h?             17
    11: 'SGDClassifier',                # HPO on serial: 1min to 5min             # 50h?                  # 36min on laptop. by default it is actually implementing an SVM https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    12: 'LinearSVC',                    # HPO on serial: 2min to 10min            # 80h?             # AttributeError: 'LinearSVC' object has no attribute 'predict_proba'
    13: 'LogisticRegression',           # HPO on serial: 1min to 1min
    14: 'RidgeClassifier',              # HPO on serial: 1min to 1min  # ok ridge is deleting features using l2 regularization but for PDC all features are dependent and deleting them does not really help in the prediction
    15: 'GaussianProcessClassifier',    # HPO on serial: 15min to 1hour        disabled because too much ram needed
    # HPO on serial: 2min to 15min  # The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples.
    16: 'SVC',
    # 17: 'MultinomialNB',  # todo fix: ValueError: Negative values in data passed to MultinomialNB (input X)
}
args['data'] = args.get('data')
args['n_jobs'] = args.get('n_jobs')
if args['iter'] is not None:
    args['model'] = [str(args['iter'] // DATASET_SIZE_PAPER)]
    args['data'] = [args['iter'] % DATASET_SIZE_PAPER]

cmd_dataset_id = args['data']

if args['model'] is None:
    args['model'] = [
        'DecisionTreeClassifier',
    ]
args['model'] = [classifier_dict[int(m)] if m.isdigit() else m for m in args['model']]
classifier_names = args['model']
assert all(m in classifier_dict.values() for m in classifier_names), f'Unknown classifier: {classifier_names}'


classifier_cpu_usage = {  # None==full parallelism  # CPU    RAM without paralelism
    'DecisionTreeClassifier': 1.,  # 13 mahiech paralelisable, 3maltha *10 w najem nzidha*3
    'ExtraTreeClassifier': 1.,
    'HistGradientBoostingClassifier': None,  # 100  paralelilisable without having the param n_jobs=-1
    'RandomForestClassifier': None,  # 100 700M  paralelilisable if n_jobs=-1
    'MLPClassifier': 1.,  # 50
    'BaggingClassifier': None,  # 100 w 100M
    'KNeighborsClassifier': 8 * .6,  # 100 w 400M
    'GradientBoostingClassifier': 1.,  # 12
    'ExtraTreesClassifier': 1.,  # 100
    'BernoulliNB': 1.,  # 17
    'GaussianNB': 1.,  # 16 and it has only one hyperparam
    'LogisticRegression': 1.,  # 500M
}
import os
num_cpus = os.cpu_count()
os.environ['LOKY_MAX_CPU_COUNT'] = str(num_cpus)

incompatible_classifiers = [
    'GaussianNB',
    'KNeighborsClassifier',
    'BernoulliNB',
    'MultinomialNB',
    'LinearSVC',
    'GaussianProcessClassifier',
    'LogisticRegression',
    'SVC',
    'SGDClassifier',
    'RidgeClassifier',
    'MLPClassifier',  # With default param PDC is better but with HPO it is not. in theory it is not compatible
]

from sklearn.metrics._scorer import _SCORERS



def get_scoring_dict(number_classes=2, model=None) -> dict[str, callable]:
    """20% Slower but with calibration scores"""
    scorer_dict = {'f1_macro': _SCORERS['f1_macro']}
    if model is not None and hasattr(model, 'predict_proba'):
        scorer_dict['neg_log_loss'] = _SCORERS['neg_log_loss']
        if number_classes == 2:
            scorer_dict['neg_brier_score'] = _SCORERS['neg_brier_score']
            scorer_dict['neg_brier_score'] = _SCORERS['neg_brier_score']
    return scorer_dict





def get_optimal_n_jobs(classifier_name, _max=None):
    if classifier_cpu_usage.get(classifier_name) is None:
        return None  # no parallelism needed
    n_jobs = int(num_cpus // classifier_cpu_usage[classifier_name])
    if _max is not None:
        n_jobs = int(min(n_jobs, int(_max)))
    if HYPER_PARAM_OPT and not isWindows:
        if classifier_name == 'RandomForestClassifier':
            n_jobs = n_jobs // 2
    return n_jobs


regressor_names = [
    'DecisionTreeRegressor'
]

estimators = {
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'ExtraTreeClassifier': ExtraTreeClassifier,
    'MLPClassifier': MLPClassifier,
    'BaggingClassifier': BaggingClassifier,
    'KNeighborsClassifier': sklearn.neighbors.KNeighborsClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'BernoulliNB': BernoulliNB,
    'GaussianNB': GaussianNB,
    'LogisticRegression': LogisticRegression,
    'SVC': SVC,
    'MultinomialNB': MultinomialNB,
    'LinearSVC': LinearSVC,
    'GaussianProcessClassifier': GaussianProcessClassifier,
    'SGDClassifier': SGDClassifier,
    'RidgeClassifier': RidgeClassifier,
}

if unknown := [clf for clf in classifier_names if clf not in estimators.keys()]:
    raise ValueError(f'Unknown classifiers: {unknown}')


wrong_classification_datasets = list({
    40589, 41496, 41707, 41713, 41716, 41719, 41720, 41721, 41722, 41724, 41725, 41726, 41730, 41731, 41733, 41737, 41740, 41744, 41746, 41750, 41753,
    41754, 41756, 41760, 41764, 41765, 41766, 41769, 41783, 41784, 41785, 41788, 41790, 41791, 41797, 41801, 41803, 41811, 41812, 41815, 41818, 41821,
    41826, 41829, 41830, 41834, 41839, 41841, 41844, 41846, 41847, 41848, 41849, 41857, 41858, 41863, 41864, 41866, 41867, 41868, 41872, 41874, 41875,
    41882, 41894, 41897, 43147, 43148, 43149, 43160, 43325, 43327, 44315, 44242, 44272, 44276, 373, 43051,
})   # because they have special datatypes
wrong_regression_datasets = list({
    45653, 45652, 45650, 45651, 710, 45589, 45588, 45591, 1228, 45761, 43477, 43944, 43452, 43686, 45061, 41943, 43403, 43483, 43582, 43384, 44203,
    43682, 43672, 43466, 43442,  # The target has too few unique values 2
    1097, 703, 45053, 45054, 45055, 45056, 45052,  # too_few_unique_3
    692, 1028,  # too_few_unique_4
    44252, 44280, 43389, 43127, 43785,  # contains_csr_matrix
    # Data IDs to remove to eliminate hash duplicates
    41517, 41519, 43123, 41518, 43959, 41516, 43943, 41515, 44957, 1027, 43440, 44223, 44187, 44191, 44192, 541, 44970, 43962, 44960,
    43056, 42436, 42437, 42438, 42439, 42444, 42445, 42464, 42900,  # I don t remeber it was mixed reasons from old code -.-'
})  # because they have special datatypes or not enough unique y values


def load_datasets(task='classification', from_cache=True, size=6000):
    assert task in ['classification', 'regression']
    if from_cache and os.path.exists(f'./benchmark/{task}_datasets.parquet'):
        datasets = pd.read_parquet(f'./benchmark/{task}_datasets.parquet').sort_values(['pairwise_complexity'], ascending=True)
    else:
        datasets = openml.datasets.list_datasets(status='active',
                                                 size=size,
                                                 number_classes=0 if task == 'regression' else None,
                                                 output_format='dataframe')
        datasets.rename(columns={"did": "data_id"}, inplace=True)
        if 'MinorityClassSize' in datasets.columns:
            datasets['MinorityClassRatio'] = datasets.MinorityClassSize / datasets.NumberOfInstances
        # https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/
        datasets['complexity'] = datasets.NumberOfFeatures**.83 * datasets.NumberOfInstances**1.11
        datasets['pairwise_complexity'] = (3 * datasets.NumberOfFeatures)**.83 * (datasets.NumberOfInstances ** 2)**1.11
        datasets['NumberOfInstances2'] = datasets.NumberOfInstances ** 2
        datasets.sort_values(['pairwise_complexity'], ascending=True, inplace=True)
        datasets.to_parquet(f'{task}_datasets.parquet')

    wrong_datasets = list(set(wrong_classification_datasets).union(wrong_regression_datasets).intersection(set(datasets.data_id)))
    datasets.index = datasets.data_id.rename('data id')
    datasets.drop(wrong_datasets, inplace=True)
    return datasets



try:
    sys.getwindowsversion()
except AttributeError:
    isWindows = False
else:
    isWindows = True

logic_processors = effective_n_jobs(n_jobs=-1)
process = psutil.Process()


def get_ram_max():
    if isWindows:
        r = process.memory_info().peak_wset
    else:
        r = process.memory_info().text
    return int(r / (1024 ** 2))  # Mb


def checkpoint(results: list, path='./tmp', step=1 if HYPER_PARAM_OPT else 10) -> None:
    # pdc_time is only the fit time per fold so real time is pdc_time * 10 * 5 * 1.5
    if not HYPER_PARAM_OPT and results[-1].get('pdc_time', 0) < 3. and len(results) % step == 0:
        return
    df = pd.DataFrame(results)
    save_results(df, path)
    print('\nCheckpoint:\n', df.mean(numeric_only=True, skipna=True), flush=True)


def convert_cv_result(cross_validate_result: dict[str, np.ndarray], prefix='base_') -> dict:
    if 'estimator' in cross_validate_result.keys():
        cross_validate_result['best_params'] = [estimator.best_params_ if hasattr(
            estimator, 'best_params_') else None for estimator in cross_validate_result['estimator']]
        cross_validate_result['cv_results'] = [estimator.cv_results_ if hasattr(
            estimator, 'cv_results_') else None for estimator in cross_validate_result['estimator']]
        del cross_validate_result['estimator']
    result = {f'{prefix}{key}': val for key, val in cross_validate_result.items()}
    result[f'{prefix}test_f1'] = result[f'{prefix}test_f1_macro'].mean()
    result[f'{prefix}time'] = cross_validate_result['fit_time'].mean()
    if prefix == 'pdc_':
        result['pdc_ram_max'] = get_ram_max()
    return result


def optimize(estimator, n_job=-1, using_mpi=False):
    if not HYPER_PARAM_OPT:
        return estimator

    if estimator.__class__.__name__ != 'PairwiseDifferenceClassifier':
        search_space = classifier_config_dict[estimator.__class__.__name__]
    else:
        search_space = pairwise_classifier_config_dict[estimator.estimator.__class__.__name__]
    return GridSearchCV(estimator, param_grid=search_space, cv=INNER_FOLDS, n_jobs=n_job, error_score='raise', verbose=0 if estimator.__class__.__name__ != 'PairwiseDifferenceClassifier' else 1)  # sklearn


import threading
import time
import psutil

class MonitorCPUUsage:
    _records: list

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_cpu_usage)
        self._records = []

    def _monitor_cpu_usage(self):
        while not self._stop_event.is_set():
            self._records.append(psutil.cpu_percent(interval=0.02, percpu=False))
            time.sleep(0.02)
            if len(self._records) > 100:
                self._records = [sum(self._records) / len(self._records)]

    def start(self):
        self._records = []
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        if self._records:
            return sum(self._records) / len(self._records)
        else:
            return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_results(df, path):
    try:
        if 'base_best_params' in df.columns:
            df['base_best_params'] = df['base_best_params'].apply(json.dumps)
        if 'pdc_best_params' in df.columns:
            df['pdc_best_params'] = df['pdc_best_params'].apply(json.dumps)
        if 'base_cv_results' in df.columns:
            df['base_cv_results'] = df['base_cv_results'].map(lambda x: json.dumps(x, cls=NumpyEncoder))
        if 'pdc_cv_results' in df.columns:
            df['pdc_cv_results'] = df['pdc_cv_results'].map(lambda x: json.dumps(x, cls=NumpyEncoder))
        df.sort_values('data_id').to_parquet(path)
    except Exception as e:
        print(f'Error in save_results: {e}')
        df.sort_values('data_id').to_parquet(path.replace('parquet', 'fastparquet'), engine='fastparquet')


def group_features(features: dict[int, openml.datasets.data_feature.OpenMLDataFeature], x_columns):
    nominal_features = []
    numeric_features = []
    string_features = []
    for openml_feature in features.values():
        if openml_feature.name not in x_columns:
            continue
        if openml_feature.data_type == 'nominal':
            nominal_features.append(openml_feature.name)
        elif openml_feature.data_type == 'numeric':
            numeric_features.append(openml_feature.name)
        elif openml_feature.data_type == 'string':
            string_features.append(openml_feature.name)
        else:
            print(f'In group_features(): unknown data type {openml_feature.data_type}')
    return numeric_features, nominal_features, string_features


def get_generic_preprocessing(numeric_features, nominal_features, string_features) -> ColumnTransformer:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

    # Create transformers for numeric and string columns
    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#standardscaler
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    string_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
    ordinal_transformer = Pipeline(steps=[('OrdinalEncoder', OrdinalEncoder())])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('str', string_transformer, string_features),
            ('ord', ordinal_transformer, nominal_features),
        ],
        remainder='passthrough',  # Pass through other columns
    )
    return preprocessor


def cast_uint(X: pd.DataFrame, y: pd.Series = None):
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].astype('float32')
    if y is not None:
        y = y.astype('float32')
    return X, y


def k_fold_split(df: pd.DataFrame, k=5) -> pd.Series:
    """
     Stratified k fold split
    :param df: all columns will be used for the stratified split
    :param k: number of folds
    :return:
    """
    from skmultilearn.model_selection import IterativeStratification
    if df.isna().sum().sum() > 0:
        print('Warning: found some nans during train test split')
        _df = df.fillna('nan')
    else:
        _df = df

    # karim is not sure that the following 2 lines are important
    _df = _df.astype('category')
    _df = _df.apply(lambda x: x.cat.codes)  # convert into numerical

    fold = pd.Series(index=df.index, dtype='int32')
    stratifier = IterativeStratification(n_splits=k, order=2)  # not sure if order should be 2 or k
    X = np.zeros(len(df))
    #
    stratifier_generator = stratifier.split(X, y=_df.values)
    for i, (_, y_idx) in enumerate(stratifier_generator):
        fold.iloc[y_idx] = i
    return fold


def describe_datasets(df):
    return df[['NumberOfFeatures', 'NumberOfSymbolicFeatures', 'NumberOfInstances', 'NumberOfInstances2', 'NumberOfClasses', 'MinorityClassRatio',]].describe().T[['min', 'max']].round(2)


def seed_everything(seed: int):
    """ Set a seed for all used libraries at once. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
