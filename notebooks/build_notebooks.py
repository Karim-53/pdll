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
         f'labels are known, and the answers are combined. With n training rows the model can learn from up to n² pairs, which '
         f'is why it is aimed at small tables. [`pdll`](https://github.com/Karim-53/pdll) wraps any scikit-learn model with the usual '
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


TABMINI_DATASETS = """[
    'parity5', 'analcatdata_fraud', 'analcatdata_aids', 'analcatdata_bankruptcy', 'analcatdata_japansolvent', 'labor',
    'analcatdata_asbestos', 'lupus', 'postoperative_patient_data', 'analcatdata_cyyoung9302', 'analcatdata_cyyoung8092',
    'analcatdata_creditscore', 'appendicitis', 'molecular_biology_promoters', 'analcatdata_boxing1', 'mux6',
    'analcatdata_boxing2', 'hepatitis', 'corral', 'glass2', 'backache', 'prnn_crabs', 'sonar', 'biomed', 'prnn_synth',
    'analcatdata_lawsuit', 'spect', 'heart_statlog', 'breast_cancer', 'heart_h', 'hungarian', 'cleve', 'heart_c',
    'haberman', 'bupa', 'spectf', 'ionosphere', 'colic', 'horse_colic', 'house_votes_84', 'vote', 'saheart', 'clean1', 'irish',
]"""


def kaggle_tabmini():
    return [
        new_markdown_cell(
            '# Pairwise Difference Learning: 2nd of 11 on TabMini\n\n'
            '**On 44 small binary datasets (32 to 500 rows), with every model at its default settings, PDL ranks 2nd of 11 '
            'classifiers by mean rank of per-dataset ROC AUC: behind CatBoost, ahead of the other nine, random forest, XGBoost '
            'and LightGBM included.** It is in the top 3 on 23 of the 44 datasets, more often than any other model (CatBoost: '
            '22). This is a placing, not a win: PDL\'s differences from CatBoost, random forest, extra trees, SVM, logistic '
            'regression and XGBoost are not statistically significant (see below).\n\n'
            'Many real projects start with a small table: a few hundred rows, a binary label, no time to tune. This notebook '
            'runs the [TabMini](https://github.com/RicardoKnauer/TabMini) benchmark ([Knauer et al. 2024]'
            '(https://arxiv.org/abs/2409.01635)) with its own protocol.\n\n'
            '*Runs on Kaggle with Internet on: the first cell installs `pdll` from PyPI, and the 44 datasets are read from PMLB '
            'on GitHub.*\n\n' + INTRO),
        new_code_cell('!pip install -q pdll==1.0.0 scikit-learn==1.9.1 catboost==1.2.10 lightgbm==4.7.0 xgboost==3.4.1'),
        new_code_cell("""import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import pdll
from pdll import PairwiseDifferenceClassifier

warnings.simplefilter('ignore')
N_JOBS = os.cpu_count()
import catboost, lightgbm, sklearn, xgboost
print('pdll', pdll.__version__, '| scikit-learn', sklearn.__version__, '| catboost', catboost.__version__,
      '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__)"""),
        new_markdown_cell(
            '## The data: TabMini, exactly as published\n\n'
            'TabMini reads each dataset from [PMLB](https://github.com/EpistasisLab/pmlb), drops rows with missing values, '
            'shuffles the rows with seed 42 and takes the largest label as the positive class. We read the files from the PMLB '
            'commit TabMini used (9 datasets were renamed later). All features are numeric.'),
        new_code_cell(f"""DATASETS = {TABMINI_DATASETS}
PMLB = 'https://github.com/EpistasisLab/pmlb/raw/872ec0a1f99ca0b55a8c8c5491da09a7da66ce13/datasets/{{0}}/{{0}}.tsv.gz'


def load(name):
    frame = pd.read_csv(PMLB.format(name), sep='\\t', compression='gzip').dropna()
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    return frame.drop(columns='target'), (frame['target'] == frame['target'].max()).astype(int).to_numpy()


data = {{name: load(name) for name in DATASETS}}
sizes = pd.Series({{name: len(y) for name, (X, y) in data.items()}})
print(f'{{len(data)}} datasets, {{sizes.min()}} to {{sizes.max()}} rows (median {{int(sizes.median())}})')"""),
        new_markdown_cell(
            '## The 11 classifiers, all at their defaults\n\n'
            'Only `random_state=0`, the thread count, `max_iter=1000` for logistic regression (the default often stops before '
            'converging) and `probability=True` for the SVM (to get probabilities) are set. Linear, k-NN and SVM models get '
            'standardized features. PDL is `PairwiseDifferenceClassifier(random_state=0)`: its default base model is an '
            'extra-trees forest with `class_weight=\'balanced\'`.'),
        new_code_cell("""def scaled(model):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)


MODELS = {
    'PDL (pdll)': lambda: PairwiseDifferenceClassifier(random_state=0),
    'CatBoost': lambda: CatBoostClassifier(random_seed=0, thread_count=N_JOBS, verbose=0, allow_writing_files=False),
    'Random forest': lambda: RandomForestClassifier(random_state=0, n_jobs=N_JOBS),
    'Extra trees': lambda: ExtraTreesClassifier(random_state=0, n_jobs=N_JOBS),
    'SVM (RBF)': lambda: scaled(SVC(probability=True, random_state=0)),
    'Logistic regression': lambda: scaled(LogisticRegression(max_iter=1000)),
    'XGBoost': lambda: XGBClassifier(random_state=0, n_jobs=N_JOBS),
    'LightGBM': lambda: LGBMClassifier(random_state=0, n_jobs=N_JOBS, verbose=-1),
    'HistGradientBoosting': lambda: HistGradientBoostingClassifier(random_state=0),
    'k-NN': lambda: scaled(KNeighborsClassifier(n_jobs=N_JOBS)),
    'Decision tree': lambda: DecisionTreeClassifier(random_state=0),
}"""),
        new_markdown_cell(
            '## Run: TabMini\'s protocol\n\n'
            'Stratified 3-fold cross-validation on the shuffled rows, ROC AUC on each test fold, averaged over the 3 folds. '
            'This takes about 17 minutes on a 12-thread laptop and up to about an hour on Kaggle\'s 4 CPUs, CatBoost and PDL '
            'taking most of it.'),
        new_code_cell("""auc = {}
for name, (X, y) in data.items():
    for model_name, make in MODELS.items():
        scores = []
        for train, test in StratifiedKFold(3).split(X, y):
            model = make().fit(X.iloc[train], y[train])
            scores.append(roc_auc_score(y[test], model.predict_proba(X.iloc[test])[:, 1]))
        auc[name, model_name] = np.mean(scores)
    print(name, end=', ', flush=True)
auc = pd.Series(auc).unstack()[list(MODELS)].round(5)   # 5 decimals, as in the benchmark repo's results
print(f'\\n{auc.shape[0]} datasets x {auc.shape[1]} classifiers scored')"""),
        new_markdown_cell('## Ranking\n\nOn each dataset the models are ranked by ROC AUC (1 = best, ties averaged); then the ranks are averaged '
                          'over the 44 datasets.'),
        new_code_cell("""ranks = auc.apply(lambda row: pd.Series(rankdata(-row.to_numpy(), method='average'), index=row.index), axis=1)
table = pd.DataFrame({'mean rank': ranks.mean(), 'mean ROC AUC': auc.mean(), 'top-3 finishes': (ranks <= 3).sum()})
table = table.sort_values('mean rank')
table.index.name = 'model'
table.round(3)"""),
        new_code_cell("""order = table.index[::-1]
colors = ['#2a78d6' if m == 'PDL (pdll)' else '#8f8d88' for m in order]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(order, table.loc[order, 'mean rank'], color=colors)
for i, m in enumerate(order):
    ax.text(table.loc[m, 'mean rank'] + 0.08, i, f"{table.loc[m, 'mean rank']:.2f}", va='center')
ax.set_xlabel('mean rank of ROC AUC over 44 datasets (lower is better)')
ax.set_title('TabMini: 44 binary datasets, 32-500 rows, default settings')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()"""),
        new_markdown_cell(
            '## Is the difference real?\n\n'
            'A Friedman test checks that the models differ at all; then PDL is compared with each model by a Wilcoxon '
            'signed-rank test over the 44 datasets, Holm-corrected for 10 comparisons.'),
        new_code_cell("""print('Friedman p =', f'{friedmanchisquare(*[auc[m] for m in auc.columns]).pvalue:.1e}')

others = [m for m in table.index if m != 'PDL (pdll)']
p_raw = np.array([wilcoxon(auc['PDL (pdll)'] - auc[m], zero_method='zsplit').pvalue for m in others])
order_p = np.argsort(p_raw)
p_holm = np.empty_like(p_raw)
p_holm[order_p] = np.maximum.accumulate(np.minimum(1, (len(p_raw) - np.arange(len(p_raw))) * p_raw[order_p]))
diff = auc[['PDL (pdll)']].to_numpy() - auc[others]
pd.DataFrame({'PDL wins': (diff > 0.005).sum(), 'ties (within 0.005)': (diff.abs() <= 0.005).sum(),
              'PDL losses': (diff < -0.005).sum(), 'mean AUC difference': diff.mean().round(4),
              'Holm p': [f'{p:.2g}' for p in p_holm]}, index=others)"""),
        new_markdown_cell(
            'PDL\'s differences from CatBoost, random forest, extra trees, SVM, logistic regression and XGBoost are not '
            'statistically significant (CatBoost: Holm p = 0.83): on 44 small datasets the top group is close. By this test '
            'PDL is significantly better than LightGBM, HistGradientBoosting, k-NN and a single decision tree; part of the lead '
            'over LightGBM and HistGradientBoosting comes from the 6 smallest datasets, where their defaults cannot split (see '
            'Scope). Read the result as "a top-group model at default settings", not as "the winner".'),
        new_markdown_cell(
            '## Robustness\n\n'
            'Four pairs of TabMini datasets are the same UCI problem, and 14 datasets had been used while developing PDL (the '
            'paper and pdll\'s own tests). The benchmark repo also finds PDL 2nd on the 25 datasets TabMini now keeps, on the '
            '23 outside TabPFN\'s prior and when any one or two datasets are left out, and in the top 3 in 98% of bootstrap '
            'resamples of the datasets. Here, the place holds without the duplicates and without the 14:'),
        new_code_cell("""def pdl_place(names):
    means = ranks.loc[names].mean().sort_values()
    return list(means.index).index('PDL (pdll)') + 1, f"{means['PDL (pdll)']:.2f}", means.index[0]

views = {
    'all 44': DATASETS,
    '40 distinct (one of each duplicate pair)': [n for n in DATASETS if n not in {'house_votes_84', 'hungarian', 'heart_c', 'horse_colic'}],
    'the 30 PDL had never been run on': [n for n in DATASETS if n not in {
        'molecular_biology_promoters', 'mux6', 'analcatdata_boxing2', 'glass2', 'prnn_synth', 'spect', 'heart_statlog',
        'heart_h', 'hungarian', 'cleve', 'heart_c', 'haberman', 'spectf', 'ionosphere'}],
}
pd.DataFrame({view: pdl_place(names) for view, names in views.items()}, index=['PDL place', 'PDL mean rank', '1st']).T"""),
        new_markdown_cell(
            '## Scope\n\n'
            '- Binary classification, at most 500 rows; one shuffle seed, 3 folds, one model seed; no model tuned.\n'
            '- ROC AUC, binary targets and small data were chosen after an earlier benchmark (35 OpenML tasks with 500 to 2,000 '
            'rows), on which PDL did better by ROC AUC than by accuracy or macro-F1, and better on binary and smaller tasks. '
            'On TabMini, PDL is 3rd by accuracy too (benchmark repo; not computed here).\n'
            '- By mean AUC instead of mean rank, PDL is 3rd.\n'
            '- Default LightGBM and HistGradientBoosting cannot split on the 6 smallest datasets (at most 57 rows) and score 0.5 '
            'there, which accounts for part of PDL\'s lead over them.\n'
            '- TabPFN and AutoML tools are not in this comparison.\n\n'
            'Full harness, per-fold scores and checks: [pdll/benchmark/tabmini](https://github.com/Karim-53/pdll/tree/main/benchmark/tabmini).'),
        new_markdown_cell('## Try it on your own small table'),
        new_code_cell("""from sklearn.model_selection import cross_val_score

X, y = data['sonar']   # your table here: a DataFrame (string columns are ordinal-encoded) or an array, and labels
model = PairwiseDifferenceClassifier(random_state=0)   # or wrap any classifier: PairwiseDifferenceClassifier(RandomForestClassifier(class_weight='balanced'))
print('ROC AUC', cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean().round(3))"""),
        new_markdown_cell(CITE),
    ]


def kaggle_pima():
    return [
        new_markdown_cell(
            '# Pima Diabetes: 11 untuned classifiers by ROC AUC\n\n'
            '**On the 768 rows of the Pima Indians Diabetes data, with the same 25 cross-validation splits for every model and '
            'nothing tuned, Pairwise Difference Learning (PDL) has the highest mean ROC AUC of the 11 (0.834).** Logistic '
            'regression and CatBoost are within 0.002, so the top three are level; PDL scores higher than random forest on 20 '
            'of the 25 folds, and higher than extra trees, LightGBM and XGBoost on all 25. On 44 small datasets '
            '([TabMini](https://github.com/Karim-53/pdll/tree/main/benchmark/tabmini)) PDL ranks 2nd of 11, behind CatBoost.\n\n'
            '*Runs on Kaggle with Internet on (the first cell installs `pdll` from PyPI) and the dataset '
            '`uciml/pima-indians-diabetes-database` attached.*\n\n' + INTRO),
        new_code_cell('!pip install -q pdll==1.0.0 scikit-learn==1.9.1 catboost==1.2.10 lightgbm==4.7.0 xgboost==3.4.1'),
        new_code_cell("""import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import pdll
from pdll import PairwiseDifferenceClassifier

warnings.simplefilter('ignore')
N_JOBS = os.cpu_count()
DATA_DIR = os.environ.get('DATA_DIR', '/kaggle/input/pima-indians-diabetes-database')
if not os.path.exists(f'{DATA_DIR}/diabetes.csv'):   # dataset not attached: fetch it from Kaggle
    import kagglehub
    DATA_DIR = kagglehub.dataset_download('uciml/pima-indians-diabetes-database')
data = pd.read_csv(f'{DATA_DIR}/diabetes.csv')
X, y = data.drop(columns='Outcome'), data['Outcome']
import catboost, lightgbm, sklearn, xgboost
print('pdll', pdll.__version__, '| scikit-learn', sklearn.__version__, '| catboost', catboost.__version__,
      '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__)
print(X.shape[0], 'rows,', X.shape[1], 'features | diabetic:', f'{y.mean():.0%}')"""),
        new_markdown_cell(
            '## The 11 classifiers, all at their defaults\n\n'
            'Only `random_state=0`, the thread count, `max_iter=1000` for logistic regression and `probability=True` for the SVM '
            'are set. Linear, k-NN and SVM models get standardized features. PDL is `PairwiseDifferenceClassifier(random_state=0)`: '
            'its default base model is an extra-trees forest with `class_weight=\'balanced\'`.'),
        new_code_cell("""def scaled(model):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)


MODELS = {
    'PDL (pdll)': PairwiseDifferenceClassifier(random_state=0),
    'Logistic regression': scaled(LogisticRegression(max_iter=1000)),
    'CatBoost': CatBoostClassifier(random_seed=0, thread_count=N_JOBS, verbose=0, allow_writing_files=False),
    'SVM (RBF)': scaled(SVC(probability=True, random_state=0)),
    'Random forest': RandomForestClassifier(random_state=0, n_jobs=N_JOBS),
    'Extra trees': ExtraTreesClassifier(random_state=0, n_jobs=N_JOBS),
    'XGBoost': XGBClassifier(random_state=0, n_jobs=N_JOBS),
    'LightGBM': LGBMClassifier(random_state=0, n_jobs=N_JOBS, verbose=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=0),
    'k-NN': scaled(KNeighborsClassifier(n_jobs=N_JOBS)),
    'Decision tree': DecisionTreeClassifier(random_state=0),
}"""),
        new_markdown_cell('## Same 25 train/test splits for every model\n\n5-fold stratified cross-validation repeated 5 times '
                          '(about 4 minutes on a 12-thread laptop; expect 10 to 15 minutes on Kaggle\'s 4 CPUs).'),
        new_code_cell("""cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
scores = {name: cross_val_score(model, X, y, cv=cv, scoring='roc_auc') for name, model in MODELS.items()}
scores = pd.DataFrame(scores)
table = pd.DataFrame({'mean ROC AUC': scores.mean(), 'std over 25 folds': scores.std(),
                      'folds where PDL is higher (of 25)': ['' if m == 'PDL (pdll)' else (scores['PDL (pdll)'] > scores[m]).sum()
                                                            for m in scores.columns]})
table = table.sort_values('mean ROC AUC', ascending=False)
table.round(4)"""),
        new_code_cell("""order = table.index[::-1]
colors = ['#2a78d6' if m == 'PDL (pdll)' else '#8f8d88' for m in order]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.grid(axis='x', color='#e6e4df', linewidth=0.8)
ax.scatter(table.loc[order, 'mean ROC AUC'], order, color=colors, s=60, zorder=3)
for i, m in enumerate(order):
    ax.text(table.loc[m, 'mean ROC AUC'] + 0.004, i, f"{table.loc[m, 'mean ROC AUC']:.3f}", va='center')
ax.set_xlim(0.65, 0.87)
ax.set_xlabel('mean ROC AUC, 5x5-fold CV (higher is better)')
ax.set_title('Pima Indians Diabetes: 11 classifiers at default settings')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.show()"""),
        new_markdown_cell(
            '**Reading the table.** PDL has the highest mean ROC AUC (0.834), but its lead over logistic regression and '
            'CatBoost is about 0.002 and it scores higher on only 14 and 13 of the 25 folds, so treat the top three as tied. '
            'Against the other tree ensembles the gap is larger: PDL scores higher than random forest on 20 of the 25 folds, '
            'and higher than extra trees, LightGBM and XGBoost on all 25 (the folds come from 5 repeats of the same data, so '
            'these counts show consistency, not significance).\n\n'
            'One dataset is one dataset. For the broader picture, see [TabMini](https://github.com/Karim-53/pdll/tree/main/benchmark/tabmini): '
            '44 small binary datasets (32 to 500 rows), every one reported, where PDL at its default settings ranks 2nd of 11 by '
            'mean rank of ROC AUC, behind CatBoost, with the top group statistically close.'),
        new_markdown_cell(
            '## Scope\n\n'
            '- One dataset, 768 rows; one CV seed (25 folds); no model tuned.\n'
            '- The data are used as published: zeros that stand for missing values (374 in Insulin, 227 in SkinThickness, 35 '
            'in BloodPressure, 11 in BMI, 5 in Glucose) are left as they are, for every model.\n'
            '- TabPFN and AutoML tools are not in this comparison.'),
        new_markdown_cell('## Use it\n\nPDL is a scikit-learn estimator, so it drops into pipelines, `cross_val_score` and '
                          '`GridSearchCV` like any classifier. Five patients it has never seen:'),
        new_code_cell("""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
model = PairwiseDifferenceClassifier(random_state=0)   # or wrap any classifier: PairwiseDifferenceClassifier(RandomForestClassifier(class_weight='balanced'))
model.fit(X_train, y_train)
model.predict_proba(X_test.head())[:, 1].round(3)      # probability of diabetes"""),
        new_markdown_cell(CITE),
    ]


NOTEBOOKS = {'kaggle_titanic.ipynb': kaggle_titanic, 'kaggle_house_prices.ipynb': kaggle_house_prices,
             'colab_quickstart.ipynb': colab_quickstart, 'kaggle_tabmini.ipynb': kaggle_tabmini,
             'kaggle_pima_diabetes.ipynb': kaggle_pima}

if __name__ == '__main__':
    for name, cells in NOTEBOOKS.items():
        notebook = new_notebook(cells=cells(), metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
        nbformat.write(notebook, os.path.join(HERE, name))
        print('wrote', name)
