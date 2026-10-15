"""Decision boundaries of 10 scikit-learn classifiers, alone and wrapped in PDL, on three 2D toy datasets.

Reproduces the README figure (results/2d_datasets.png). Needs matplotlib: ``pip install pdll matplotlib``.
Adapted from the script in PR #6 (scikit-learn's "Classifier comparison" example).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pdll import PairwiseDifferenceClassifier


def base_classifiers():
    return [
        KNeighborsClassifier(3),
        SVC(kernel='linear', C=0.025, random_state=42, probability=True),
        SVC(gamma=2, C=1, random_state=42, probability=True),
        HistGradientBoostingClassifier(max_depth=5, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        # pair features [a, b, a - b] are collinear by construction: QDA needs a little regularisation
        QuadraticDiscriminantAnalysis(reg_param=0.1),
    ]


def datasets():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    X += 2 * np.random.RandomState(2).uniform(size=X.shape)
    return [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1), (X, y)]


def main(out='2d_datasets.png'):
    names = [type(clf).__name__ for clf in base_classifiers()]
    rows = [(data, wrap) for data in datasets() for wrap in (False, True)]
    figure, axes = plt.subplots(len(rows), len(names) + 1, figsize=(27, 3 * len(rows)))
    cm, cm_bright = plt.cm.RdBu, ListedColormap(['#FF0000', '#0000FF'])
    for row, ((X, y), wrap) in enumerate(rows):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        classifiers = [PairwiseDifferenceClassifier(clf, random_state=0) if wrap else clf for clf in base_classifiers()]
        panels = [(None, 'Input data')] + list(zip(classifiers, names))
        for col, (clf, name) in enumerate(panels):
            ax = axes[row, col]
            if clf is not None:
                model = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)
                DecisionBoundaryDisplay.from_estimator(model, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5, grid_resolution=40)
                ax.text(x_max - 0.3, y_min + 0.3, f'{model.score(X_test, y_test):.2f}'.lstrip('0'), size=15, horizontalalignment='right')
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xticks=(), yticks=())
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel('PDC(classifier)' if wrap else 'classifier alone', size=14)
    figure.suptitle('Decision boundaries: each classifier alone vs. wrapped in PairwiseDifferenceClassifier', fontsize=20)
    plt.tight_layout()
    plt.savefig(out)
    print('saved', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
