"""Generate the pdll showcase notebooks (Kaggle, Zindi, Colab) from the cell lists below.

    python notebooks/build_notebooks.py            # writes notebooks/*.ipynb

Each notebook reads its data from DATA_DIR (default: the platform's input folder), so the same file runs on Kaggle and,
for testing, locally with DATA_DIR pointing to a copy of the data.
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER = ('[Belaid, Rabus & Hüllermeier, *Pairwise Difference Learning for Classification*, Discovery Science 2024]'
         '(https://doi.org/10.1007/978-3-031-78980-9_18)')
INTRO = (f'**Pairwise Difference Learning (PDL)** trains a model on *pairs* of rows: "do these two rows have the same label?" '
         f'(or, for regression, "how far apart are their targets?"). A new row is then compared with the training rows, whose '
         f'labels are known, and the answers are combined. With n training rows the model sees up to n² examples, which helps '
         f'most when data is scarce. [`pdll`](https://github.com/Karim-53/pdll) wraps any scikit-learn model with the usual '
         f'`fit` / `predict` API. Method: {PAPER}. *Disclosure: I am one of the authors.*')
CITE = ('If PDL is useful in your work, please cite the paper:\n\n```bibtex\n@inproceedings{belaid2024pairwise,\n'
        '  title     = {Pairwise Difference Learning for Classification},\n'
        '  author    = {Belaid, Mohamed Karim and Rabus, Maximilian and H{\\"u}llermeier, Eyke},\n'
        '  booktitle = {Discovery Science},\n  pages     = {284--299},\n  publisher = {Springer},\n  year      = {2024},\n'
        '  doi       = {10.1007/978-3-031-78980-9_18}\n}\n```')


def kaggle_titanic():
    return [
        new_markdown_cell('# Titanic with Pairwise Difference Learning: raw DataFrame in, no feature engineering\n\n' + INTRO),
        new_code_cell('!pip install -q -U pdll'),
        new_code_cell("""import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import pdll
from pdll import PairwiseDifferenceClassifier

DATA_DIR = os.environ.get('DATA_DIR', '/kaggle/input/titanic')
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
print('pdll', pdll.__version__, '| train', train.shape, '| test', test.shape)"""),
        new_markdown_cell('Drop the identifiers and keep everything else as it is: strings (`Sex`, `Embarked`) and missing values '
                          '(`Age`, `Fare`, `Embarked`). Category columns are understood by both the base model and pdll.'),
        new_code_cell("""features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def prepare(df):
    X = df[features].copy()
    for column in ('Sex', 'Embarked'):
        X[column] = X[column].astype('category')
    return X

X, y, X_test = prepare(train), train['Survived'], prepare(test)
X.head()"""),
        new_markdown_cell('Same base model alone and wrapped in PDL, same 5-fold cross-validation. HistGradientBoosting handles '
                          'the missing values natively.'),
        new_code_cell("""base = HistGradientBoostingClassifier(class_weight='balanced', categorical_features='from_dtype', random_state=0)
pdl = PairwiseDifferenceClassifier(base, random_state=0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for name, model in [('HistGradientBoosting', base), ('PDL(HistGradientBoosting)', pdl)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'{name:28} accuracy {scores.mean():.3f} ± {scores.std():.3f}')"""),
        new_markdown_cell('Fit on all training passengers and write the submission.'),
        new_code_cell("""pdl.fit(X, y)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pdl.predict(X_test)})
submission.to_csv('submission.csv', index=False)
submission.head()"""),
        new_markdown_cell('PDL also tells you how much the training passengers *disagree* about a prediction: '
                          '`predict_proba_samples` gives the posterior from each training passenger (anchor) separately.'),
        new_code_cell("""per_anchor = pdl.predict_proba_samples(X_test[:5])[:, :, 1]   # P(survived) according to each anchor
pd.DataFrame({'P(survived)': pdl.predict_proba(X_test[:5])[:, 1].round(2),
              'spread across anchors': per_anchor.std(axis=1).round(2)}, index=test['PassengerId'][:5])"""),
        new_markdown_cell(CITE),
    ]


def kaggle_house_prices():
    return [
        new_markdown_cell('# House Prices with Pairwise Difference Regression: raw DataFrame in\n\n' + INTRO),
        new_code_cell('!pip install -q -U pdll'),
        new_code_cell("""import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

import pdll
from pdll import PairwiseDifferenceRegressor

DATA_DIR = os.environ.get('DATA_DIR', '/kaggle/input/house-prices-advanced-regression-techniques')
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
print('pdll', pdll.__version__, '| train', train.shape, '| test', test.shape)"""),
        new_markdown_cell('All 79 columns as they come, strings and missing values included. The competition scores the RMSE '
                          'of log(SalePrice), so we model the log price directly.'),
        new_code_cell("""both = pd.concat([train, test]).drop(columns=['Id', 'SalePrice'])
for column in both.select_dtypes(exclude='number').columns:   # strings -> categories (same codes in train and test)
    both[column] = both[column].astype('category')
X, X_test = both.iloc[:len(train)], both.iloc[len(train):]
y = np.log1p(train['SalePrice'])
print(f'{X.shape[1]} columns, {sum(X.dtypes == "category")} categorical, {int(X.isna().sum().sum())} missing values')"""),
        new_code_cell("""base = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=0)
pdl = PairwiseDifferenceRegressor(base, random_state=0)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
for name, model in [('HistGradientBoosting', base), ('PDL(HistGradientBoosting)', pdl)]:
    rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    print(f'{name:28} RMSE(log price) {rmse.mean():.4f} ± {rmse.std():.4f}')"""),
        new_code_cell("""pdl.fit(X, y)
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': np.expm1(pdl.predict(X_test))})
submission.to_csv('submission.csv', index=False)
submission.head()"""),
        new_markdown_cell(CITE),
    ]


def colab_quickstart():
    return [
        new_markdown_cell('# pdll quickstart: Pairwise Difference Learning in five minutes\n\n' + INTRO),
        new_code_cell('!pip install -q -U pdll'),
        new_markdown_cell('**Wrap any scikit-learn model.** A single decision tree on 300 handwritten digits, alone and wrapped in '
                          'PDL (same shuffled 5-fold cross-validation for both):'),
        new_code_cell("""import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from pdll import PairwiseDifferenceClassifier

X, y = load_digits(return_X_y=True)
subset = np.random.RandomState(0).permutation(len(y))[:300]      # a small dataset
X, y = X[subset], y[subset]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
tree = DecisionTreeClassifier(random_state=0)
print('decision tree      :', cross_val_score(tree, X, y, cv=cv).mean().round(3))
print('PDL(decision tree) :', cross_val_score(PairwiseDifferenceClassifier(tree), X, y, cv=cv).mean().round(3))"""),
        new_markdown_cell('**Raw DataFrames** work as they are: strings, categories and missing values (Titanic from OpenML).'),
        new_code_cell("""from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)
X = X[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]    # drop identifiers such as name or ticket

hgb = HistGradientBoostingClassifier(class_weight='balanced', categorical_features='from_dtype', random_state=0)
print('HistGradientBoosting      :', cross_val_score(hgb, X, y, cv=cv).mean().round(3))
print('PDL(HistGradientBoosting) :', cross_val_score(PairwiseDifferenceClassifier(hgb, random_state=0), X, y, cv=cv).mean().round(3))"""),
        new_markdown_cell('PDL does not win everywhere: it helps tree models most, on small to medium tabular data. See the '
                          '[benchmark on 130 real datasets](https://github.com/Karim-53/pdll/tree/main/benchmark/real_world).'),
        new_markdown_cell(CITE),
    ]


NOTEBOOKS = {'kaggle_titanic.ipynb': kaggle_titanic, 'kaggle_house_prices.ipynb': kaggle_house_prices,
             'colab_quickstart.ipynb': colab_quickstart}

if __name__ == '__main__':
    for name, cells in NOTEBOOKS.items():
        notebook = new_notebook(cells=cells(), metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
        nbformat.write(notebook, os.path.join(HERE, name))
        print('wrote', name)
