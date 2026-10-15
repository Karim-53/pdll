"""Quickstart: wrap a scikit-learn classifier with PDL and compare it with the classifier alone."""
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from pdll import PairwiseDifferenceClassifier


def main():
    X, y = load_wine(return_X_y=True)

    rf = RandomForestClassifier(class_weight='balanced', random_state=0)
    pdc = PairwiseDifferenceClassifier(estimator=rf)

    print('RandomForest      accuracy:', cross_val_score(rf, X, y, cv=5).mean().round(3))
    print('PDC(RandomForest) accuracy:', cross_val_score(pdc, X, y, cv=5).mean().round(3))

    pdc.fit(X, y)
    print('probabilities of the first sample:', pdc.predict_proba(X[:1]).round(3))


if __name__ == '__main__':
    main()
