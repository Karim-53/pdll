"""Per-anchor posteriors: how much the anchors (training points) disagree about a prediction."""
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

from pdll import PairwiseDifferenceClassifier


def main():
    X, y = load_iris(return_X_y=True)
    pdc = PairwiseDifferenceClassifier(ExtraTreesClassifier(class_weight='balanced', random_state=0), use_prior=True).fit(X[::2], y[::2])

    samples = pdc.predict_proba_samples(X[1::50])  # (n_samples, n_anchors, n_classes)
    print('mean posterior :', samples.mean(axis=1).round(3))
    print('spread (std)   :', samples.std(axis=1).round(3))


if __name__ == '__main__':
    main()
