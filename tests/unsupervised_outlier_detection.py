# Data manipulation and analysis
import pandas as pd
import numpy as np


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import sem, ttest_rel
# Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from pdll._pairwise import PairwiseDifferenceOutlierDetection

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')



#  Dictionary of dataset file paths
file_paths = {
    1: "/datasets/celeba_baldvsnonbald_normalised.csv",
    2: "/datasets/census-income-full-mixed-binarized.csv",
}

all_results = []

#  Iterate over all datasets
for idx, path in file_paths.items():
    dataset_name = f"Dataset {idx}"
    print(f" Dataset {idx}: {path}")

    df = pd.read_csv(path)

    # Preprocessing
    X = df.drop(columns=['class'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_0 = df[df['class'] == 0]
    df_1 = df[df['class'] == 1]

    # Class balancing (different strategy depending on dataset)
    if path in ["census-income-full-mixed-binarized.csv", "UNSW_NB15_traintest_backdoor.csv", "celeba_baldvsnonbald_normalised.csv"]:
        df_0 = df_0.sample(1000)   # Random sampling without random_state
        df_1 = df_1.sample(70)
    else:
        df_0 = df_0.sample(3000)
        df_1 = df_1.sample(200)

    df_sampled = pd.concat([df_0, df_1])

    X_sampled = df_sampled.drop(columns=['class']).values
    y_sampled = df_sampled['class'].values

    # Train/Test Split (without random_state, fully random split)
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.4)
    assert len(np.unique(y_train)) == 2
    assert len(np.unique(y_test)) == 2

    # Baseline: IsolationForest
    iso_forest = IsolationForest()
    iso_forest.fit(X_train)
    y_pred_iso = np.where(iso_forest.predict(X_test) == -1, 1, 0)

    # Metrics for IsolationForest
    f1_iso = f1_score(y_test, y_pred_iso, average='macro')
    auc_roc_iso = roc_auc_score(y_test, y_pred_iso)
    auc_pr_iso = average_precision_score(y_test, y_pred_iso)

    all_results.append({
        "Dataset": dataset_name,
        "Method": "IsolationForest",
        "Percentile": "baseline",
        "F1_macro": f1_iso,
        "AUC_ROC": auc_roc_iso,
        "AUC_PR": auc_pr_iso
    })

    # Train PairwiseDifferenceOutlierDetection
    model = PairwiseDifferenceOutlierDetection()

    # >>> Threshold is optimized inside 'fit()' based on mean anomaly scores
    # >>> Threshold is found on X_train_pair with best F1 macro (based on y_train)
    model.fit(pd.DataFrame(X_train), y_train)   # Important "i did it already in PairwiseDifferenceOutlierDetection": Pass X_train_Pair as DataFrame

    # Predict using the best found threshold 
    # >>> Prediction is made on X_test by applying the selected best threshold from training
    y_pred_pdl = model.predict(pd.DataFrame(X_test))

    # Metrics for PairwiseDifferenceOutlierDetection
    f1_pdl = f1_score(y_test, y_pred_pdl, average='macro')
    auc_roc_pdl = roc_auc_score(y_test, y_pred_pdl)
    auc_pr_pdl = average_precision_score(y_test, y_pred_pdl)

    all_results.append({
        "Dataset": dataset_name,
        "Method": "PairwiseDifferenceOutlierDetection",
        "Percentile": model.percentile_,  # The best percentile chosen during training
        "F1_macro": f1_pdl,
        "AUC_ROC": auc_roc_pdl,
        "AUC_PR": auc_pr_pdl
    })

# Save all results
df_results = pd.DataFrame(all_results)
print("Summary:")
print(df_results)