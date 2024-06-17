# import tpot
# tpot_classifier_config_dict = tpot.config.classifier.classifier_config_dict
# assert effective_n_jobs(n_jobs=-1) > 1, effective_n_jobs(n_jobs=-1)
# classifier_config_dict = {key.split('.')[-1]: val for key, val in tpot_classifier_config_dict.items()}

min_samples_split = [2, 4, 8, 16, 21]  # start with the default of RandomForestClassifier
min_samples_leaf = [1, 2, 4, 10, 21]  # start with the default of RandomForestClassifier
classifier_config_dict = {}
classifier_config_dict['DecisionTreeClassifier'] = {'criterion': ['gini', 'entropy'],
                                                    'max_depth': [None, 1, 2, 4, 6, 8, 11],
                                                    'min_samples_split': min_samples_split,
                                                    'min_samples_leaf': min_samples_leaf}
classifier_config_dict['RandomForestClassifier'] = {'n_estimators': [100],  # default ok
                                                    'criterion': ['gini', 'entropy'],
                                                    'min_samples_split': min_samples_split,
                                                    'max_features': ['sqrt', .05, .17, .29, .41, .52, .64, .76, .88, 1.],
                                                    'min_samples_leaf': min_samples_leaf,
                                                    'bootstrap': [True, False],
                                                    'n_jobs': [-1]}  # it is not exploiting entire CPUs :(
classifier_config_dict['ExtraTreeClassifier'] = {'criterion': ['gini', 'entropy'],
                                                 # 'max_depth': [None, 3, 32],
                                                 'min_samples_split': [2, 5, 10],
                                                 'min_samples_leaf': [1, 2, 4],
                                                 # one of the 3 important hyperparameters according to the original paper
                                                 'max_features': ['sqrt', 'log2', None],
                                                 # one of the 3 important hyperparameters according to the original paper
                                                 'max_leaf_nodes': [None, 2, 12, 56],
                                                 'min_impurity_decrease': [0., 0.1, 0.5]}
classifier_config_dict['HistGradientBoostingClassifier'] = {'max_iter': [100, 10, ],
                                                            'learning_rate': (0.1, 0.01, 1),
                                                            # 'max_depth': [None, 3, 13],
                                                            'max_leaf_nodes': [31, 3, 256],  # None
                                                            'min_samples_leaf': [20, 4, 64],
                                                            # default=20  The minimum number of samples per leaf. For small datasets with less than a few hundred samples, it is recommended to lower this value since only very shallow trees would be built.
                                                            'l2_regularization': [0, 0.01, 0.1],
                                                            'max_bins': [255, 2, 64],
                                                            # 'early_stopping': [True, False],  # default ‘auto’, early stopping is enabled if the sample size is larger than 10000. If True, early stopping is enabled, otherwise early stopping is disabled.
                                                            }
classifier_config_dict['BaggingClassifier'] = {'n_estimators': [10, 5, 100, 256],
                                               'max_samples': [1., .5],
                                               'max_features': [0.5, 0.9, 1.0],
                                               'bootstrap': [True, False],
                                               'bootstrap_features': [False, True],
                                               'n_jobs': [-1]}
classifier_config_dict['ExtraTreesClassifier'] = {'n_estimators': [100],
                                                  'criterion': ['gini', 'entropy'],
                                                  'max_features': ['sqrt', 0.05, 0.17, 0.29, 0.41, 0.52, 0.64, 0.76, 0.88, 1.],
                                                  'min_samples_split': min_samples_split,
                                                  'min_samples_leaf': min_samples_leaf,
                                                  'bootstrap': [False, True],
                                                  'n_jobs': [-1]}
classifier_config_dict['GradientBoostingClassifier'] = {'learning_rate': (0.1, 0.01, 1),
                                                        # 'max_depth': [3, 6, None],
                                                        'min_samples_split': min_samples_split,
                                                        'min_samples_leaf': min_samples_leaf,
                                                        'subsample': [1., 0.05, 0.37, 0.68, ],
                                                        'max_features': [None, 0.15, 0.68]}  # If None, then max_features=n_features
classifier_config_dict['MLPClassifier'] = {'alpha': [0.0001, 0.001, 0.01, 0.1],
                                           'learning_rate_init': [0.001, 0.01, 0.1, 0.5, 1.0]}

classifier_config_dict['KNeighborsClassifier'] = {'n_neighbors': [5, 1, 2, 3, 7, 17],
                                                  'weights': ['uniform', 'distance'],
                                                  'p': [2, 1],
                                                  'n_jobs': [-1]}  # n_jobs Doesn’t affect fit method.
classifier_config_dict['GaussianNB'] = {}
classifier_config_dict['BernoulliNB'] = {'alpha': [1.0, 0.001, 0.01, 0.1, 10.0, 100.0],
                                         'fit_prior': [True, False]}
classifier_config_dict['SGDClassifier'] = {'loss': ['hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
                                           # sklearn.utils._param_validation.InvalidParameterError: The 'loss' parameter of SGDClassifier must be a str among {'log_loss', 'perceptron', 'squared_error', 'modified_huber', 'epsilon_insensitive', 'huber', 'squared_epsilon_insensitive', 'squared_hinge', 'hinge'}. Got 'log' instead.
                                           # ‘hinge’ gives a linear SVM. no need it here
                                           'penalty': ['l2', 'elasticnet'],
                                           # ValueError: alpha must be > 0 since learning_rate is 'optimal'. alpha is used to compute the optimal learning rate.
                                           'alpha': [.0001, 0.01, ],
                                           # 'learning_rate': ['invscaling', 'constant'],  # default=’optimal’
                                           # 'fit_intercept': [True, False],
                                           'l1_ratio': [.15, 0.0, 1.0, 0.75, 0.5],
                                           # 'eta0': [0.01, 0.1, 1.0],  # The default value is 0.0 as eta0 is not used by the default schedule learning_rate='optimal'.
                                           'power_t': [0.5, 0.0, 10.0, 0.1, 100.]}
classifier_config_dict['LinearSVC'] = {
    # 'penalty': ['l1', 'l2'],
    # 'loss': ['hinge', 'squared_hinge'], ValueError: Unsupported set of arguments: The combination of penalty='l2' and loss='hinge' are not supported when dual=False, Parameters: penalty='l2', loss='hinge', dual=False
    # 'dual': [True, False],  # ValueError: Unsupported set of arguments: The combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True
    # The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.
    'dual': ['auto'],  # "auto" option is added in version 1.3 and will be the default in version 1.5.
    'tol': [1e-04, 1e-05, 1e-03, 1e-02, 1e-01],
    'C': [1., 0.0001, 0.001, 0.01, 0.1, 0.5, 5.0, 10.0, 15.0, 20.0, 25.0],
}
classifier_config_dict['LogisticRegression'] = {
    # 'penalty': ['l1', 'l2'],  # Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.
    'C': [1., 0.0001, 0.001, 0.01, 0.1, 0.5, 5.0, 10.0, 15.0, 20.0, 25.0],
    # 'dual': [True, False],  # Solver lbfgs supports only dual=False, got dual=True
}
classifier_config_dict['RidgeClassifier'] = {'alpha': [1., 0.1, 10.]}
classifier_config_dict['SVC'] = {
    'C': [1., 0.0001, 0.001, 0.01, 0.1, 0.5, 5.0, 10.0, 15.0, 20.0, 25.0],
    'tol': [1e-03, 1e-05, 1e-04, 1e-02, 1e-01],
}
classifier_config_dict['GaussianProcessClassifier'] = {'copy_X_train': [False],
                                                       'random_state': [9],
                                                       'n_jobs': [-1]}
classifier_config_dict['MultinomialNB'] = {'alpha': [1.0, 0.001, 0.01, 0.1, 10.0],
                                           'fit_prior': [True, False]}


# Models the periodicity
# seasonal_kernel = (
#     2.0**2
#     * RBF(length_scale=100.0, length_scale_bounds=(1e-2,1e7))
#     * ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-2,1e7),
#                      periodicity=1.0, periodicity_bounds="fixed")
# )
# # Models small variations
# irregularities_kernel = ConstantKernel(constant_value=1,constant_value_bounds =(1e3,1e6)) * RationalQuadratic(length_scale=1.0,
#                                 length_scale_bounds=(1e-2,1e7), alpha=1.0)
# # Models noise
# noise_kernel = 0.1**2 * RBF(length_scale=0.1, length_scale_bounds=(1e-2,1e7)) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))
# co2_kernel = (seasonal_kernel + irregularities_kernel + noise_kernel)
# classifier_config_dict['GaussianProcessClassifier'] = {'kernel': [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(),],
#                                                        'copy_X_train': [False], 'n_jobs': [None],
#                                                        # 1*WhiteKernel(), I think it doesn't make sense to just model noise...
#                                                        # 1*RBF() + 1*DotProduct() + 1*Matern() + 1*RationalQuadratic() + 1*WhiteKernel(),
#                                                        # co2_kernel,
#                                                        }
# todo add test to check that these parameters are correct


def prefix_estimator(dico: dict) -> dict:
    return {f'estimator__{param}': search_space for param, search_space in dico.items()}


pairwise_classifier_config_dict = {key: prefix_estimator(val) for key, val in classifier_config_dict.items()}

# SGDClassifier hpo too slow

if __name__ == '__main__':
    from sklearn.model_selection import ParameterGrid
    meta = []
    for key, config_dict in pairwise_classifier_config_dict.items():
        l = len(ParameterGrid(config_dict))
        print(key, l)
        assert l <= 1000, f"Too many parameters for {key}, {l}"
        meta.append({'name': key, '#parameters': len(config_dict), '#combinations': l})
        if l < 56:
            print(f'{key} has only {l} < 56 combinations. It will not take advantage of the parallelism.')
    from pairwise.benchmark_utils import classifier_dict, incompatible_classifiers

    missing_hpo = set(classifier_dict.values()) - (set(classifier_config_dict.keys()))
    assert len(missing_hpo) == 0, f"Missing hpo for {missing_hpo}"
    import pandas as pd
    df = pd.DataFrame(meta)
    print(df)

    print(df[~df.name.isin(incompatible_classifiers)].to_latex(index=False,
                                                               caption='Description of the search space per estimator.',
                                                               label='tab:describe_hyper_space',
                                                               escape=False,
                                                               na_rep='',
                                                               )
          )

    params = pd.DataFrame([{'estimator': estimator, 'parameter': parameter, 'values': str(values)[1:-1].replace("'", '')}
                           for estimator, c in classifier_config_dict.items()
                           for parameter, values in c.items()
                           if estimator not in incompatible_classifiers and len(values) > 1]).set_index('estimator', append=True,).swaplevel()
    print(params.to_latex(index=True,
                          caption='Search space per estimator.',
                          label='tab:hyper_space',
                          escape=False,
                          na_rep='',
                          ).replace('_', ' ').replace(r'\multirow[t]{4}{*}', '').replace(r'\multirow[t]{5}{*}', '').replace(r'\multirow[t]{6}{*}', '')
          )
