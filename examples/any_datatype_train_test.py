"""Train Pairwise with a non-numeric dataset."""
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.extend('../')
from pdll import PairwiseDifferenceClassifier, PDCDataTransformer

import numpy as np
from sklearn.datasets import fetch_openml


def test_any_dataset():
    np.random.seed(53)  # Set the random seed for reproducibility

    # Load the Adult Census dataset from OpenML
    adult_census = fetch_openml(name="adult", version=2, as_frame=True)

    # Access the data and target
    X = adult_census.data  # Features
    y = adult_census.target  # Target (e.g., income as a binary variable)

    # to have a quick run, we subsample the data
    X = X.sample(frac=0.01)
    y = y.loc[X.index]

    print(X.head(3))
    #     age  workclass    fnlwgt  ... capital-loss  hours-per-week native-country
    # 0  25.0    Private  226802.0  ...          0.0            40.0  United-States
    # 1  38.0    Private   89814.0  ...          0.0            50.0  United-States
    # 2  28.0  Local-gov  336951.0  ...          0.0            40.0  United-States
    print('X contains non-numeric features and must be converted before using PairwiseDifference')

    print(y.head(3))
    # 0    <=50K
    # 1    <=50K
    # 2     >50K
    # Name: class, dtype: category
    # Categories (2, object): ['<=50K', '>50K']
    print('Same for the target y, it must be converted before using PairwiseDifference')

    classifier = PairwiseDifferenceClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'))
    # classifier.fit(X, y)  # This would rise an error because PairwiseDifference cannot handle non-numeric features. Please use the pipeline below:

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    nominal_features = ['workclass', 'education', 'marital-status', 'relationship']  #  'occupation',
    string_features = ['race', 'sex'] # 'native-country'
    X = X[numeric_features + nominal_features + string_features]
    data_transformer = PDCDataTransformer(numeric_features, nominal_features, string_features, y_type='string')
    ml_pipeline = Pipeline(steps=[
        ('pre-processor', data_transformer),
        ('model', classifier),
    ])

    print('fitting...')
    ml_pipeline.fit(X, y)
    print('score:', ml_pipeline.score(X, y))

    y_pred = ml_pipeline.predict(X)
    proba_pred = ml_pipeline.predict_proba(X)

    assert ml_pipeline.score(X, y) == 1.0


if __name__ == "__main__":
    test_any_dataset()
