"""PDL on a raw DataFrame: strings, categories and missing values, no preprocessing (downloads Titanic from OpenML)."""
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from pdll import PairwiseDifferenceClassifier


def main():
    try:
        X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)
    except TypeError:  # scikit-learn's pandas ARFF parser fails with pandas >= 3.1; its other parser does not
        X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True, parser='liac-arff')
    X = X[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].iloc[:600]  # drop identifiers; keep the demo fast
    y = y.iloc[:600]
    print('column dtypes:', X.dtypes.astype(str).value_counts().to_dict(), '| missing values:', int(X.isna().sum().sum()))

    # HistGradientBoosting handles the missing ages natively; pdll encodes the string and category columns itself.
    pdc = PairwiseDifferenceClassifier(HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=0))
    print('PDC(HistGradientBoosting) accuracy:', cross_val_score(pdc, X, y, cv=5).mean().round(3))


if __name__ == '__main__':
    main()
