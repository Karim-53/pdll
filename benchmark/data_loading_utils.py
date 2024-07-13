import numpy as np
import pandas as pd
import warnings
import hashlib
import openml
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from tqdm import tqdm

from benchmark.benchmark_utils import load_datasets, cast_uint, group_features, get_generic_preprocessing

NR_DATASETS = 150


def get_regression_datasets(number_instances_min=25, number_instances_max=2100, cmd_dataset_id=None) -> pd.DataFrame:
    """
    Look at the available datasets and select the ones that are suitable for the task.
    :param number_instances_min: Minimum number of instances in the dataset int(K / (1 - 1 / NR_OF_SPLITS_CV) / (1 - VALIDATION_RATIO))
    :param number_instances_max: Maximum number of instances in the dataset
    :param search_size: Number of datasets to search through
    :return: Dataframe containing the selected datasets
    """
    # print("Select datasets...")
    datasets = load_datasets(task='regression', from_cache=True)
    selected_datasets = datasets[
        (datasets.NumberOfMissingValues == 0) &
        (datasets.NumberOfInstances > number_instances_min) &
        (datasets.NumberOfInstances <= number_instances_max) &
        (datasets.NumberOfFeatures >= 3) & (datasets.NumberOfFeatures <= 100) &
        (datasets.complexity <= 1e5) &
        ((datasets.NumberOfInstances <= 1000) | (datasets.NumberOfFeatures <= 20)) &
        (datasets.format != 'Sparse_ARFF')
    ].sort_values(['NumberOfInstances', 'NumberOfFeatures'])
    # selected_datasets = selected_datasets.iloc[:NR_DATASETS]

    print('Selected', len(selected_datasets), 'datasets out of', len(datasets), 'found.')
    if cmd_dataset_id is not None:
        print('Selected dataset nb', cmd_dataset_id, 'of size\n', selected_datasets.iloc[cmd_dataset_id][['NumberOfInstances', 'NumberOfFeatures']])
        return selected_datasets.iloc[cmd_dataset_id]
    return selected_datasets


def group_features_for_regression(features: dict[int, openml.datasets.data_feature.OpenMLDataFeature], x_columns):
    """
    Group the features in the dataset into nominal, numeric and string features.
    :param features: Dict of features (the features are the values)
    :param x_columns:
    :return: Three lists of feature names => numeric_features, nominal_features, string_features
    """
    nominal_features = []
    numeric_features = []
    string_features = []
    for openml_feature in features.values():
        if openml_feature.name not in x_columns:
            warnings.warn(f"Some features name found that wasn't in the dataframe! => '{openml_feature.name}'")
            continue
        if openml_feature.data_type == 'nominal':
            nominal_features.append(openml_feature.name)
        elif openml_feature.data_type == 'numeric':
            numeric_features.append(openml_feature.name)
        elif openml_feature.data_type == 'string':
            string_features.append(openml_feature.name)
        else:
            print(f"Unknown data type '{openml_feature.data_type}'")
    assert len(nominal_features) + len(numeric_features) + len(string_features) >= 1, "No features found!"
    return numeric_features, nominal_features, string_features


def get_generic_preprocessing_for_regression(numeric_features, nominal_features, string_features) -> ColumnTransformer:
    """
    Create a generic preprocessor pipeline for the dataset. This pipeline can be used to preprocess the data before
    training a model. Scale numeric features and one-hot-encode nominal features.
    :param numeric_features: List of numeric feature names
    :param nominal_features: List of nominal feature names
    :param string_features: List of string feature names
    :return:
    """
    # Create transformers for numeric and string columns
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    string_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
    ordinal_transformer = Pipeline(steps=[('OrdinalEncoder', OrdinalEncoder())])

    # Combine transformers using ColumnTransformer
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('str', string_transformer, string_features),
            # todo add something for dates, dates are probably strings
            ('ord', ordinal_transformer, nominal_features),
        ],
        remainder='passthrough',  # Pass through other columns
    )


def cast_uint_to_float(X: pd.DataFrame, y: pd.Series = None):
    """ Cast all columns that are of type uint to float. """
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].astype('float32')
    if y is not None:
        y = y.astype(np.float32)
    return X, y


def hash_pandas_objects(X, y):
    """
    Hash a pandas DataFrame and Series together to a SHA-256 hash.
    """
    # Concatenate DataFrame and Series as a string
    combined_str = X.to_string() + y.to_string()

    # Encode the combined string and create a SHA-256 hash
    hash_result = hashlib.sha256(combined_str.encode()).hexdigest()

    return hash_result


def verify_all_datasets():
    selected_datasets = get_regression_datasets(cmd_dataset_id=None)
    lista = []
    for data_id in tqdm(selected_datasets.data_id):
        X, y = get_processed_regression_dataset(data_id)
        if X is None:
            print(data_id, ',')
            continue
        hash = hash_pandas_objects(X, y)
        lista.append({
            'data_id': data_id,
            'hash': hash,
            'y_truth_range': y.max() - y.min(),
        })
    df = pd.DataFrame(lista)
    # List to store data_ids to remove
    to_remove = []
    # Iterate over each group
    for hash_value, group in df.groupby('hash'):
        if len(group) > 1:
            # Add all but one data_id to the removal list
            to_remove.extend(group['data_id'].iloc[:-1].tolist())

    assert len(to_remove) == 0, f"Data IDs to remove to eliminate hash duplicates: {to_remove}"
    return df


def get_processed_regression_dataset(data_id):
    dataset = openml.datasets.get_dataset(int(data_id), download_qualities=True, download_features_meta_data=False, download_data=True)
    if dataset.qualities['NumberOfInstances'] > 11000:
        print('error in data ID:', data_id, '\t', 'Dataset is too large and would slow the experiment')
        return None, None

    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

    if y.dtype in ['object', 'category']:  # Skip if y is not numeric
        # y = y.astype('category').cat.codes.astype(np.float32)
        print(f'error in data ID: {data_id}\t(Classification dataset, #classes={y.nunique()}.) '
              'Skipping the dataset...')
        return None, None
    if any(pd.isna(y)):  # Skip if NaN
        print(f'error in data ID:{data_id}\t(Found some y=NaN.) Skipping the dataset')
        return None, None
    # Data processing

    X, y = cast_uint(X, y)
    numeric_features, nominal_features, string_features = group_features(dataset.features, X.columns)
    preprocessing = get_generic_preprocessing(numeric_features, nominal_features, string_features)
    X = pd.DataFrame(preprocessing.fit_transform(X))
    if any(isinstance(e, csr_matrix) for e in X.values.flatten()):
        print('error in data ID:', data_id, '\t', 'X contains csr_matrix')
        return None, None
    X = X.dropna(axis=1, how='all')  # Drop columns with all NaN values
    X = X.astype(np.float32)

    if len(X.columns) == 0:
        return None, None
    if X.isna().any().any():
        print('error in data ID:', data_id, '\t', 'Some features are NaNs in the X set')
        return None, None
    if any(x in pd.Series(X.values.flatten()).apply(type).unique() for x in ('csr_matrix', 'date',)):  # todo think about adding  'str'
        print('error in data ID:', data_id, '\t', 'Dataset contains sparse data')
        return None, None
    non_numeric_columns = X.select_dtypes(exclude='number').columns
    if len(non_numeric_columns) > 0:
        print('error in data ID:', data_id, '\t', 'non-numeric columns found:', non_numeric_columns)
        return None, None
    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#standardscaler
    y = pd.Series(StandardScaler().fit_transform(pd.DataFrame(y)).flatten(), index=y.index)
    y = y.astype(np.float32)
    y.name = 'target'
    if y.nunique() < 5:
        print('error in data ID:', data_id, '\t', 'The target has too few unique values', y.nunique())
        return None, None

    return X, y
