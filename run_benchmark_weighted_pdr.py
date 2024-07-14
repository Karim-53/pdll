"""
Basic script to reproduce the results of the paper
"Weighted Pairwise Difference Learning", Belaid et al., 2024.

You need to specify the regressor and the dataset to evaluate as arguments
when running the script. The script will evaluate the performance of the
regressor on the dataset using cross-validation and save the results to a
parquet file.

Example usage (random forest on small dataset):
python run_benchmark_weighted_pdr.py --regressor RandomForestRegressor --data 4

The needed time to run depends on the regressor and the dataset.
"""
import inspect
import itertools
import os
import time
import warnings
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold
from tqdm import tqdm

from benchmark.benchmark_utils import seed_everything, isWindows, args
from benchmark.data_loading_utils import get_processed_regression_dataset, get_regression_datasets
from pdll import PairwiseDifferenceRegressor

warnings.filterwarnings('ignore')

SEED = 11
NR_OF_SPLITS_CV = 5
NR_REPETITIONS = 5  # How often each experiment is repeated
VALIDATION_RATIO = 0.3  # Validation ratio of the training set => #samples * TRAIN_RATIO * VALIDATION_RATIO
K = 5  # Knn parameter

PATH = f'./results/{SEED}/'.lower()
os.makedirs(PATH, exist_ok=True)


def instantiate_base_model(model):
    """ Create a model based on the model parameter and the models supported features. """
    if 'n_jobs' in inspect.signature(model).parameters:
        return model(random_state=SEED, n_jobs=-1)  # add n_jobs for the models that support it (else ignored)
    return model(random_state=SEED)


# MAPE is not a good idea because a score could be very large (almost inf but not inf) because of one prediction dont y_truth is very close to 0
score = mean_absolute_error
# then just use the standard MAE and for the paper and stats use the relative error that I define as MARE = MAE / delta_y_truth
# delta_y_truth = np.max(y_truth) np.min(y_truth) + epsilon
# assert delta_y_truth > 0.1
# print(df.delta_y_truth.describe()# should be in the range of 0.1 to 1
# epsilon = np.finfo(np.float32).eps


def evaluate_model_on_dataset(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Evaluate the model on the dataset. Evaluates base model, standard Padre, weighted Padre, and StackedPadre.
    :param model: Model to evaluate
    :param X_train: Training set
    :param X_val: Validation set
    :param X_test: Test set
    :param y_train: Training target
    :param y_val: Validation target
    :param y_test: Test target
    :return: dict containing the results in form {method: score}
    """
    results = []

    # Baseline
    base_model = instantiate_base_model(model)
    base_model.fit(X_train, y_train)
    results.append({'method': "Baseline",
                    'val_set': "",
                    'test_error': score(y_test, base_model.predict(X_test)),
                    'val_error': np.nan,  # score(y_val, base_model.predict(X_val)),  # even if it is not used...
                    'weights': None})

    # Standard Padre
    padre = PairwiseDifferenceRegressor(instantiate_base_model(model))
    padre.fit(X_train, y_train)
    results.append({'method': "Padre",
                    'val_set': "",
                    'test_error': score(y_test, padre.predict(X_test)),
                    'val_error': np.nan,  # score(y_val, padre.predict(X_val)),  # even if it is not used...
                    'weights': None})

    # Weighted Padre (all weighting methods)
    best_weighted_score = float("inf")
    best_weighted_method = {}
    best_theoretical_weighted_score = float("inf")
    best_theoretical_weighted_method = {}
    for weighting_method in padre._name_to_method_mapping.keys():
        padre.learn_anchor_weights(X_val=X_val, y_val=y_val, X_test=X_test, method=weighting_method, enable_warnings=False)
        val_error = score(y_val, padre.predict(X_val))
        test_error = score(y_test, padre.predict(X_test))
        results.append({'method': f"Padre {weighting_method}",
                        'val_set': "validation",
                        'val_error': val_error,
                        'test_error': test_error,
                        'weights': padre.sample_weight_})
        if val_error < best_weighted_score:
            best_weighted_score = val_error
            best_weighted_method = results[-1]
        if test_error < best_theoretical_weighted_score:
            best_theoretical_weighted_score = test_error
            best_theoretical_weighted_method = results[-1]

        padre.learn_anchor_weights(X_train, y_train, method=weighting_method, enable_warnings=False)
        val_error = score(y_val, padre.predict(X_val))
        test_error = score(y_test, padre.predict(X_test))
        results.append({'method': f"Padre {weighting_method}",
                        'val_set': "train",
                        'val_error': val_error,
                        'test_error': test_error,
                        'weights': padre.sample_weight_})
        if val_error < best_weighted_score:
            best_weighted_score = val_error
            best_weighted_method = results[-1]
        if test_error < best_theoretical_weighted_score:
            best_theoretical_weighted_score = test_error
            best_theoretical_weighted_method = results[-1]

    results.append({'method': "Padre best weighted",
                    'val_set': "validation",
                    'source_method': best_weighted_method['method'],
                    'test_error': best_weighted_method['test_error'],
                    'val_error': best_weighted_method['val_error'],
                    'weights': best_weighted_method['weights']})
    results.append({'method': "Padre best theoretical weighted",
                    'val_set': "test",
                    'source_method': best_theoretical_weighted_method['method'],
                    'test_error': best_theoretical_weighted_method['test_error'],
                    'val_error': best_theoretical_weighted_method['val_error'],
                    'weights': best_theoretical_weighted_method['weights']})

    return results


def get_regressor(regressor_name):
    """ Imports are done locally because some models affect the global state of the program like the number of CPUs visible to joblib"""
    if regressor_name == 'DecisionTreeRegressor':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor
    if regressor_name == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    if regressor_name == 'ExtraTreeRegressor':
        from sklearn.tree import ExtraTreeRegressor
        return ExtraTreeRegressor
    if regressor_name == 'HistGradientBoostingRegressor':
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor
    if regressor_name == 'BaggingRegressor':
        from sklearn.ensemble import BaggingRegressor
        return BaggingRegressor
    if regressor_name == 'ExtraTreesRegressor':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor
    if regressor_name == 'GradientBoostingRegressor':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor
    if regressor_name == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        return LinearRegression


def evaluate_performance_on_dataset(regressor, data_id: int, verbose: bool = True):
    """
    Evaluate the performance of different models and method on a given dataset using cross-validation.

    :param data_id: ID of the dataset to evaluate
    :param test_ratio: Ratio of the dataset to use for testing
    :param validation_ratio: Ratio of the training set to use for validation
    :param nr_of_splits: Number of splits to use for the evaluation
    :param verbose: Whether to print the progress of the splits
    :return: List of dictionaries. Every dictionary contains the results of a single method on the dataset
    """
    start_time = time.time()

    # Load the dataset
    X, y = get_processed_regression_dataset(data_id)
    if X is None:
        return

    k_fold = RepeatedKFold(n_splits=NR_OF_SPLITS_CV, n_repeats=NR_REPETITIONS, random_state=SEED)

    n_jobs = k_fold.get_n_splits()
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    if regressor is RandomForestRegressor:
        n_jobs = 9
    if regressor is ExtraTreesRegressor:
        n_jobs = 9
    from numpy.core._exceptions import _ArrayMemoryError
    from joblib.externals.loky.process_executor import TerminatedWorkerError
    import gc
    while n_jobs >= 1:
        try:
            all_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(run_fold)(i, train_index, test_index, X, y, regressor, data_id) for i, (train_index, test_index) in enumerate(k_fold.split(X, y)))
            # Flatten the list of results
            all_results = [result for result_set in all_results for result in result_set]
            assert len(all_results) > 0
            df = pd.DataFrame(all_results)
            data_time = time.time() - start_time
            df['data_time'] = data_time
            df['y_truth_range'] = y.max() - y.min()

            print("Finished evaluation! Duration:", str(timedelta(seconds=int(data_time))))
            return df
        except (_ArrayMemoryError, TerminatedWorkerError, MemoryError):
            if n_jobs is None or n_jobs <= 1:
                raise
            n_jobs = n_jobs // 2
            print('n_jobs reduced to', n_jobs, flush=True)
            gc.collect()
            continue


def run_fold(i, train_index, test_index, X, y, regressor, data_id):
    split_nr = "split_" + str(i)
    # Shuffle and split the dataset into train and test set
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_RATIO, random_state=SEED)

    results_temp = evaluate_model_on_dataset(regressor, X_train, X_val, X_test, y_train, y_val, y_test)
    results_temp = [{
        "dataset": data_id,
        "basemodel": regressor.__name__,
        "data split nr": split_nr,
    } | v
        for v in results_temp]
    return results_temp


if __name__ == '__main__':
    seed_everything(SEED)
    verbose = False
    resume = True
    if isWindows:
        resume = False

    print(datetime.now().strftime("%Y-%m-%d %H:%M"))

    parquet_file = PATH + "regression_PDR_results.parquet"

    if os.path.exists(parquet_file):
        print('File exists:', parquet_file)
        exit(0)

    selected_datasets = get_regression_datasets()
    regressors = [
        "BaggingRegressor",
        "DecisionTreeRegressor",
        "ExtraTreeRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "RandomForestRegressor"
    ]

    print("Start evaluation...")
    list_of_dfs = []
    for regressor_name, dataset_id in tqdm(itertools.product(regressors, selected_datasets.data_id)):
        df = evaluate_performance_on_dataset(regressor=get_regressor(regressor_name), data_id=dataset_id)
        df['weights'] = df['weights'].apply(lambda x: x.tolist() if x is not None else None)

        list_of_dfs.append(df)
        df.to_parquet(parquet_file)     # Save each iteration to avoid losing data in case of crash
