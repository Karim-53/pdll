import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def create_2d_plot(appyly_pairwise_difference_learning = False):
    if appyly_pairwise_difference_learning:
        #! pip install pdll
        # from pdll import PairwiseDifferenceClassifier
        # if you clone the repo then
        from pairwise.padre import PairwiseDifferenceClassifier


    classifiers = [
        KNeighborsClassifier(3, n_jobs=-1),
        SVC(kernel="linear", C=0.025, random_state=42, probability=True),
        SVC(gamma=2, C=1, random_state=42, probability=True),
        HistGradientBoostingClassifier(max_depth=5, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42, n_jobs=-1),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    names = [clf.__class__.__name__ for clf in classifiers]

    if appyly_pairwise_difference_learning:
        classifiers = [PairwiseDifferenceClassifier(estimator=clf) for clf in classifiers]

    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    # import openml
    # pn_dataset_1 = openml.datasets.get_dataset(713)
    # pn_dataset_2 = openml.datasets.get_dataset(782)
    # pn_dataset_3 = openml.datasets.get_dataset(791)
    # pn_dataset_4 = openml.datasets.get_dataset(801)
    # pn_dataset_5 = openml.datasets.get_dataset(860)
    # pn_dataset_6 = openml.datasets.get_dataset(895)
    # dataset_7 = openml.datasets.get_dataset(464)

    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
        # pn_dataset_2,
        # dataset_7
    ]

    figure = plt.figure(figsize=(27, 10))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        if isinstance(ds, tuple):
            X, y = ds  # For local datasets
        else:
            # For OpenML datasets
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
            if 'P' in y.values and 'N' in y.values:
                X, y = X.values, np.where(y == 'P', 1, 0)
            elif '0' in y.values and '1' in y.values:
                X, y = X.values, y.values.astype(int)
            else:
                X, y = X.values, y.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print(f' clf = {clf}')
            print(f'Score = {score}')
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    if appyly_pairwise_difference_learning:
        title = 'Boundary Analysis with Pairwise Difference Classifier'
        file_name = 'boundary_analysis_pairwise_difference_classifier.png'
    else:
        title = 'Boundary Analysis without Pairwise Difference Classifier'
        file_name = 'boundary_analysis_without_pairwise_difference_classifier.png'

    figure.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()
    plt.close()

create_2d_plot(appyly_pairwise_difference_learning=False)
create_2d_plot(appyly_pairwise_difference_learning=True)
# combine the two images first above the second:
from PIL import Image

# Read the images
img1 = Image.open('boundary_analysis_pairwise_difference_classifier.png')
img2 = Image.open('boundary_analysis_without_pairwise_difference_classifier.png')

# Get the dimensions of the images
width1, height1 = img1.size
width2, height2 = img2.size

# Create a new image with the combined height
combined_height = height1 + height2
combined_img = Image.new('RGB', (max(width1, width2), combined_height))

# Paste the images one above the other
combined_img.paste(img1, (0, 0))
combined_img.paste(img2, (0, height1))

# Save the combined image
combined_img.save('2d_datasets.png')
