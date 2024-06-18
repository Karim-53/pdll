"""Pairwise idea 4: PDC for binary and multi class classification"""
import gc
from joblib.externals.loky.process_executor import TerminatedWorkerError
from numpy.core._exceptions import _ArrayMemoryError
from scipy.sparse import csr_matrix
import inspect
from datetime import timedelta
from typing import Any
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from datetime import datetime
from pdll import PairwiseDifferenceClassifier
from benchmark.benchmark_utils import *

using_mpi = False
print('Not using MPI')


def get_multi_class_datasets(number_instances_minimum=21, cmd_dataset_id=None):
    """
    :param number_instances_minimum: openml-cc18's minimum is 20
    :param cmd_dataset_id:
    :return:
    """
    datasets = load_datasets(from_cache=True)
    instances_max = 600 if HYPER_PARAM_OPT else 3000

    subsampled_datasets = datasets[
        # (datasets.NumberOfMissingValues < datasets.NumberOfInstances*datasets.NumberOfFeatures*0.05) &
        (datasets.NumberOfMissingValues == 0) &
        (datasets.NumberOfClasses >= 2) & (datasets.NumberOfClasses <= 20) &
        # (datasets.NumberOfInstances <= (600 if HYPER_PARAM_OPT else 4000)) &
        # (datasets.NumberOfInstances >= 10000) &
        (datasets.NumberOfInstances >= 2 * number_instances_minimum) & (datasets.NumberOfInstances <= instances_max) &
        # (datasets.NumberOfInstances.isin(sizes)) &
        (datasets.NumberOfFeatures >= 3) & (datasets.NumberOfFeatures <= 100) &
        # (datasets.pairwise_complexity <= (1.3e8 if HYPER_PARAM_OPT else 10.e8)) &
        (datasets.MinorityClassSize >= number_instances_minimum) &
        (datasets.MinorityClassRatio >= 0.03) &
        ((datasets.NumberOfInstances <= 1000) | (datasets.NumberOfFeatures <= 20)) &
        (datasets.format != 'Sparse_ARFF')
    ].sort_values(['NumberOfInstances', 'NumberOfFeatures'], ascending=True,)
    if HYPER_PARAM_OPT:
        subsampled_datasets = subsampled_datasets.iloc[-2 * DATASET_SIZE_PAPER::2]
    # subsampled_datasets = conditioned_datasets.sort_values(['NumberOfFeatures', 'NumberOfInstances']).groupby('n').first()
    # subsampled_datasets.index = subsampled_datasets.data_id
    # assert subsampled_datasets.NumberOfInstances.max() >= 10000, subsampled_datasets.NumberOfInstances.max()

    # big_datasets = [45553, 4538, 375, 43976, 1507, 42178, 847, 182, 28, 40499, 1525, 1489, 1460, 42192, 41146, 44160,
    #                 45039, 43925, 1558, 44, 41000, 40983, 43892, 43973, 46, 3, 40678, 44232, 1487, 44091, 45562, 36, 40999, 41007,]
    # conditioned_datasets.drop(set(big_datasets).intersection(
    #     set(conditioned_datasets.data_id)), inplace=True)  # these datasets are ok but it's just that we have enough for the experiment so we discarded some sparsly according to the nb of instances
    # current_sizes = conditioned_datasets.NumberOfInstances.values
    # sizes = np.concatenate((conditioned_datasets.NumberOfInstances.values[:DATASET_SIZE_PAPER], sizes[-1:]))
    # base = 2
    # sizes = np.logspace(start=np.log(2*number_instances_minimum)/np.log(base), stop=np.log(instances_max)/np.log(base), num=6000, base=base, dtype=int)
    # print(sum(conditioned_datasets.NumberOfInstances.isin(sizes)))
    # selected_sizes = np.concatenate([conditioned_datasets[conditioned_datasets.NumberOfInstances.isin(sizes)].NumberOfInstances.values[:DATASET_SIZE_PAPER-1],
    #                                 np.array([int(conditioned_datasets.NumberOfInstances.max())])] )
    # datasets.iloc[:DATASET_SIZE_PAPER] = datasets.iloc[:DATASET_SIZE_PAPER].sort_values(by='complexity')
    # assert len(subsampled_datasets) == DATASET_SIZE_PAPER, len(subsampled_datasets)
    print('Selected', len(subsampled_datasets), 'datasets out of', len(datasets), 'found.')
    print(describe_datasets(subsampled_datasets))
    if cmd_dataset_id is not None:
        print('Selected dataset nb', cmd_dataset_id, 'of size', subsampled_datasets.iloc[cmd_dataset_id][['NumberOfInstances', 'NumberOfFeatures']])
        return subsampled_datasets.iloc[cmd_dataset_id]
    return subsampled_datasets.sort_values(['NumberOfInstances', 'NumberOfFeatures'])


def get_processed_classification_dataset(data_id, number_classes=None):
    dataset = openml.datasets.get_dataset(data_id, download_qualities=True, download_features_meta_data=False, download_data=True)
    if dataset.qualities['NumberOfInstances'] > 11000:
        print('error in data ID:', data_id, '\t', 'Dataset is too large and would slow the experiment')
        return None, None

    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

    # Data processing
    y = y.astype('category').cat.codes.astype(np.float32)
    if number_classes is not None and y.nunique() != number_classes:
        print('error in data ID:', data_id, '\t', 'y_train do not contain all classes. expected', number_classes, ' but found', y.unique())
        return None, None

    X, _ = cast_uint(X)
    numeric_features, nominal_features, string_features = group_features(dataset.features, X.columns)
    preprosessing = get_generic_preprocessing(numeric_features, nominal_features, string_features)
    X = pd.DataFrame(preprosessing.fit_transform(X))
    if any(isinstance(e, csr_matrix) for e in X.values.flatten()):
        print('error in data ID:', data_id, '\t', 'X contains csr_matrix')
        return None, None
    X = X.dropna(axis=1, how='all')  # Drop columns with all NaN values
    X = X.astype(np.float32)

    if len(X.columns) == 0:
        return None, None
    if X.isna().any().any():
        print('error in data ID:', data_id, '\t', 'Some features are NaNs in the X set')
        return None, None
    if any(x in pd.Series(X.values.flatten()).apply(type).unique() for x in ('csr_matrix', 'date',)):
        print('error in data ID:', data_id, '\t', 'Dataset contains sparse data')
        return None, None
    return X, y


def get_path(seed=9):
    path = f'./results/'
    return path.lower()


def check_cv_split(X, y, cv, number_classes):
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        _check_split(X_train, y_train, X_test, y_test, number_classes)


def _check_split(X_train, y_train, X_test, y_test, number_classes):
    if len(X_train) < 2 * number_classes:
        print('error in data ID:', data_id, '\t', 'The training set is too small', len(X_train),
              '. Total dataset size is ', len(X))
        raise ValueError()
    if y_train.nunique() != number_classes:
        print('error in data ID:', data_id, '\t', 'y_train do not contain all classes. expected', number_classes,
              ' but found', y_train.unique())
        raise ValueError()
    if y_test.nunique() != number_classes:
        print('error in data ID:', data_id, '\t', 'y_test do not contain all classes. expected', number_classes,
              ' but found', y_test.unique())
        raise ValueError()    # Since we use f1 score it's better to have all classes


def get_base_classifier(classifier_name, seed=53, for_pdc=False):
    classifier_args = {}
    base_CLASSIFIER_CLASS = estimators[classifier_name]
    if 'random_state' in inspect.signature(base_CLASSIFIER_CLASS.__init__).parameters:
        classifier_args['random_state'] = seed
    if classifier_name in ['RandomForestClassifier', 'ExtraTreesClassifier', 'SGDClassifier']:
        classifier_args['n_jobs'] = -1
        # if args['n_jobs'] is not None:
        #     classifier_args['n_jobs'] = args['n_jobs']
        # args['verbose'] = 100
    if for_pdc:
        if 'class_weight' in inspect.signature(base_CLASSIFIER_CLASS.__init__).parameters:
            classifier_args['class_weight'] = 'balanced'
    return base_CLASSIFIER_CLASS(**classifier_args)


def quick_analysis(df, classifier_name, classifier_t0):
    # print('\n\n RQ.3: Classification relative improvement of pdc over the base:',
    #       improvement(df.improvement_pdc_test_over_base))  # 46.6 % +- 4.94
    # print('\n\n RQ.3: Classification absolute improvement of pdc over the base:', df['diff'].median(), '+-', df['diff'].sem())  # .16 % +- .013
    # print('RQ.3: Classification pairwise strict better than base in ', wins(df.pdc_test_f1 > df.base_test_f1))  # (193 / 250) 	 77 %

    # RQ.4
    # print('\n\n RQ.4 Classification relative improvement of weighted over pdc:', improvement(df.improvement_weighted_over_pdc))  # 3 +- 1
    # print('RQ.4: Classification pdc better than pdc in ', wins(df.weighted_pdc_test_f1 > df.pdc_test_f1))
    # print('\n\n RQ.4 Classification relative improvement of weighted over pdc (if improved):', improvement(
    #     df[df.improvement_weighted_over_pdc > 0].improvement_weighted_over_pdc))

    # RQ.21
    # print(f'\n\n RQ.21 mean pdc brier {df.pdc_test_brier.mean():.3g} v.s. base brier {df.base_test_brier.mean():.3g}')
    # print('RQ.21: Brier pdc better than base in ', wins(df.pdc_test_brier < df.base_test_brier))

    print('\n\n\n Other results')
    # print(df.improvement_weighted_over_pdc.describe())
    # print(df)

    lista = [method for method in ('base', 'pdc', 'eyke') if method in str(df.columns)]
    lista_f1 = [f'{method}_test_f1' for method in lista]
    best_scores = df[lista_f1].max(axis='columns')

    print(f'\n Classification Obtaining the best score, limited to {DATASET_SIZE_PAPER}:')
    for method in lista:
        print(f'{method:15}', ':', sum(df[f'{method}_test_f1'] == best_scores), '/', len(df),
              '\t\t', round(sum(df[f'{method}_test_f1'] == best_scores) / len(df) * 100), '%')

    print('time', classifier_name, f'{timedelta(seconds=time.time() - classifier_t0)}')
    # print(df.corr(numeric_only=True).round(1).dropna(axis=1, how='all').dropna(axis=0, how='all'))
    print(pd.concat([df.mean(numeric_only=True).rename('mean'), df.median(numeric_only=True).rename('median')], axis=1).sort_index())
    m = df.mean(numeric_only=True).round(3).astype(str)
    print(f"""
    F1       train   test
    base     {m.loc['base_train_f1']}    {m.loc['base_test_f1']}
    pairwise {m.loc['pdc_train_f1']}    {m.loc['pdc_test_f1']}

    """)
    # Brier    train   test
    # base     {m.loc['base_train_brier']}    {m.loc['base_test_brier']}
    # pairwise {m.loc['pdc_train_brier']}    {m.loc['pdc_test_brier']}

    # eyke     {m.loc['eyke_train_f1']}    {m.loc['eyke_test_f1']}
    # eyke     {m.loc['eyke_train_brier']}    {m.loc['eyke_test_brier']}
    # base   .68   .56
    # pair   .99   .70
    # eyke   1     .65


def get_file(classifier_name, id=cmd_dataset_id):
    file = f'{DATASET_SIZE_PAPER}_{classifier_name}'
    if id is not None:
        file = f'{classifier_name}_{id}'
    return file + '.parquet'


def safe_cross_validate(model, n_jobs=-1, param_grid=None) -> dict:
    prefix = 'base_' if not isinstance(model, PairwiseDifferenceClassifier) else 'pdc_'
    n_jobs_cv = effective_n_jobs(-1)  # cross validate will use all the cores
    if HYPER_PARAM_OPT:
        # if isinstance(model, PairwiseDifferenceClassifier):
        #     n_jobs_cv = 16  # HPO will be fully expoiting the parallelism, but spawning the small jobs is veeery slow
        # else:
        n_jobs_cv = os.cpu_count()
        if isinstance(model, PairwiseDifferenceClassifier):
            if isinstance(model.estimator, (BaggingClassifier)):
                n_jobs_cv = None  # to debug only
        # if param_grid < os.cpu_count(): # nestd jobs do not work
        #     n_jobs_cv = os.cpu_count() // param_grid # parallelism divided between cross validation and HPO
    else:
        n_jobs_cv = min(n_jobs_cv, cv.get_n_splits())
        if isinstance(model, PairwiseDifferenceClassifier):
            if isinstance(model.estimator, (BaggingClassifier)):
                if len(X) > 2300:
                    n_jobs_cv = 5
            if isinstance(model.estimator, (HistGradientBoostingClassifier, GradientBoostingClassifier)):
                if len(X) > 1900:
                    n_jobs_cv = 5
            if isinstance(model.estimator, ExtraTreeClassifier):
                if len(X) > 2500:
                    n_jobs_cv = 10
            if isinstance(model.estimator, (RandomForestClassifier, ExtraTreesClassifier)):
                if len(X) > 700:
                    n_jobs_cv = 2
            if isinstance(model.estimator, ExtraTreesClassifier):
                if len(X) > 2100:
                    n_jobs_cv = 1
            if isinstance(model.estimator, RidgeClassifier):
                if len(X) > 900:
                    n_jobs_cv = 4
            if len(X) > 6000:  # DecisionTree
                n_jobs_cv = 1  # plus ca marche pas

    model = optimize(model, n_job=effective_n_jobs(-1), using_mpi=using_mpi)  # khali n_job=-1 hetha a7sen 7all
    psutil.cpu_percent(interval=None, percpu=False)
    
    sys.stdout.flush()
    sys.stderr.flush()
    # results = cross_validate(model,
    #                          X=X.values, y=y.values, cv=cv, scoring=scoring_dict,
    #                          return_train_score=True, return_estimator=HYPER_PARAM_OPT,
    #                          n_jobs=n_jobs_cv,
    #                          error_score='raise',
    #                          verbose=1 if isinstance(model, PairwiseDifferenceClassifier) else 1,
    #                          )
    # sys.stdout.flush()
    # sys.stderr.flush()
    # gc.collect()
    # results['cpu'] = psutil.cpu_percent(interval=None, percpu=False)
    # if not isWindows:
    #     import resource
    #     results['ram'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # max ram usage in Gb
    # else:
    #     results['ram'] = np.nan
    # if results['cpu'] < 80.:
    #     print('[WARNING] CPU usage', results['cpu'], flush=True)
    # else:
    #     print('CPU usage OK:', results['cpu'], flush=True)
    # return convert_cv_result(results, prefix=prefix)
    last_exception = None
    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
    n_jobs_cv = 13
    while n_jobs_cv is None or n_jobs_cv >= 1:
        try:
            results = cross_validate(model,
                                     X=X.values, y=y.values, cv=cv, scoring=scoring_dict,
                                     return_train_score=True, return_estimator=HYPER_PARAM_OPT,
                                     n_jobs=n_jobs_cv,
                                     error_score='raise',
                                     verbose=100 if isinstance(model, PairwiseDifferenceClassifier) else 1,
                                     )
        except (_ArrayMemoryError, TerminatedWorkerError, MemoryError) as e:
            if n_jobs_cv is None or n_jobs_cv <= 1:
                raise
            last_exception = e
            n_jobs_cv = n_jobs_cv // 2
            print('n_jobs_cv reduced to', n_jobs_cv, flush=True)
            gc.collect()
            continue
        results['n_jobs_cv'] = n_jobs_cv
        results['cpu'] = psutil.cpu_percent(interval=None, percpu=False)
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()
        if not isWindows:
            import resource
            results['ram'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # max ram usage in Gb
        else:
            results['ram'] = np.nan
        if results['cpu'] < 80.:
            print('[WARNING] CPU usage', results['cpu'], flush=True)
        else:
            print('CPU usage OK:', results['cpu'], flush=True)
        return convert_cv_result(results, prefix=prefix)
    else:
        if last_exception is not None:
            raise last_exception

    # psutil.cpu_percent(interval=None, percpu=False)
    # try:
    #     with joblib.parallel_backend(backend):
    #         results = cross_validate(model,
    #                                  X=X.values, y=y.values, cv=cv, scoring=scoring_dict,
    #                                  return_train_score=True, return_estimator=True,
    #                                  n_jobs=n_jobs_cv,
    #                                  error_score='raise',
    #                                  verbose=10,
    #                                  )
    # except (_ArrayMemoryError, TerminatedWorkerError, MemoryError) as e:
    #     if backend=='dask':
    #         raise
    #     else:
    #         print('memory error, restarting with dask...')
    #         with joblib.parallel_backend('dask'):
    #             results = cross_validate(model,
    #                                      X=X.values, y=y.values, cv=cv, scoring=scoring_dict,
    #                                      return_train_score=True, return_estimator=True,
    #                                      n_jobs=n_jobs_cv,
    #                                      error_score='raise',
    #                                      verbose=10,
    #                                      )
    #
    # sys.stderr.flush()
    # gc.collect()
    # results['cpu'] = psutil.cpu_percent(interval=None, percpu=False)
    # if results['cpu'] < 80.:
    #     print('[WARNING] CPU usage', results['cpu'], flush=True)
    # else:
    #     print('CPU usage OK:', results['cpu'], flush=True)
    # return convert_cv_result(results, prefix=prefix)


if __name__ == '__main__':
    # init_mpi_workers()
    print(datetime.now().strftime("%Y-%m-%d %H:%M"))

    print('Args:', args)
    print('Goal:', classifier_names, cmd_dataset_id)

    # print('default classifier_names:', classifier_names)
    print('logic_processors n_jobs initially = ', effective_n_jobs(n_jobs=-1))
    print('os.cpu_count()', os.cpu_count())
    print('HYPER_PARAM_OPT:', HYPER_PARAM_OPT)
    print('DATASET_SIZE_PAPER:', DATASET_SIZE_PAPER)
    print('REPEATS:', REPEATS)
    # print('FOLDS:', FOLDS)
    # print('INNER_FOLDS:', INNER_FOLDS)
    # backend, backend_n_jobs = get_active_backend()
    # print(f"Current backend used: {backend}")
    # print(f"Number of jobs in the backend: {backend_n_jobs}")
    # print('default_parallel_config', default_parallel_config)
    

    resume = True
    seed = 9
    cv = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=seed)
    test_ratio = 1. / FOLDS
    # validation_ratio = 1. / INNER_FOLDS

    path = get_path(seed)
    number_instances_minimum = 40  # int(ceil(4 / (1 - test_ratio) / (1 - validation_ratio)))
    datasets = get_multi_class_datasets(number_instances_minimum=number_instances_minimum, cmd_dataset_id=cmd_dataset_id)

    t0 = time.time()
    for classifier_name in classifier_names:
        print(classifier_name, '####################################################', flush=True)
        if HYPER_PARAM_OPT and classifier_name not in classifier_config_dict:
            print('no configuration found optimising', classifier_name)
            continue
        if HYPER_PARAM_OPT:
            param_grid = len(ParameterGrid(classifier_config_dict[classifier_name]))
            print('hyper parameter grid:', param_grid)
        else:
            param_grid = None
        file = get_file(classifier_name)
        tmp_file = f'{path}tmp/{file}'
        if resume and os.path.isfile(tmp_file):
            print('resuming from', tmp_file)
            df = pd.read_parquet(tmp_file)
            results = df.to_dict('records')
            if len(results) >= DATASET_SIZE_PAPER:
                print('classifier_name already completed')
                continue
        else:
            df = None
            results = []
        classifier_t0 = time.time()
        # hyperparameter_n_jobs = None # -1 if classifier_config_dict['DecisionTreeClassifier'].get('n_jobs', None) is None else None
        pbar = tqdm(zip(datasets.data_id, datasets.NumberOfClasses), total=min(DATASET_SIZE_PAPER, len(datasets)), smoothing=.4)
        # pbar = tqdm(zip(datasets.iloc[145:].data_id, datasets.iloc[145:].NumberOfClasses), total=DATASET_SIZE_PAPER)
        # pbar = tqdm(zip([464], [2]), total=1)  # smallest
        for i, (data_id, number_classes) in enumerate(pbar):
            description = f"{classifier_name} ID:{data_id} done:{len(results)}"
            pbar.set_description(description)
            if resume and df is not None and data_id in df.data_id.values:
                continue
            small_tmp_file = f'{path}tmp/{get_file(classifier_name, id=cmd_dataset_id[0] if cmd_dataset_id is not None else None)}'
            print(small_tmp_file, os.path.isfile(small_tmp_file))
            if resume and os.path.isfile(small_tmp_file):
                small_df = pd.read_parquet(small_tmp_file)
                if small_df.data_id[0] == data_id:
                    print('resuming from', small_tmp_file)
                    results.append(small_df.to_dict('records')[0])
                    continue
            try:
                X, y = get_processed_classification_dataset(data_id, number_classes)
                if X is None:
                    continue
                check_cv_split(X, y, cv, number_classes)

                result: dict[str, Any] = {'data_id': data_id, 'target_unique_vals': y.nunique(), 'NumberOfFeatures_post_processing': X.shape[1],
                                          'pairwise_complexity_post_processing': (3 * X.shape[1])**.83 * (X.shape[0] ** 2)**1.11, }
                data_t0 = time.time()

                pbar.set_description(f"{description} base size {X.shape}")
                scoring_dict = get_scoring_dict(number_classes, model=get_base_classifier(classifier_name))
                n_jobs = get_optimal_n_jobs(classifier_name)
                result.update(safe_cross_validate(get_base_classifier(classifier_name), n_jobs, param_grid))
                # assert len(results) > 0 or not pd.isna(result['base_test_f1']), 'The first Base fit failed, probably something to debug.'

                pbar.set_description(f"{description} pdc")
                
                result.update(safe_cross_validate(PairwiseDifferenceClassifier(
                    get_base_classifier(classifier_name, for_pdc=True)), n_jobs, param_grid))
                assert len(results) > 0 or not pd.isna(result['pdc_test_f1']), 'The first PDC fit failed, probably something to debug.'
                result['data_time'] = time.time() - data_t0
            except np.core._exceptions._ArrayMemoryError as e:
                print('error in data ID:', data_id, '\t _ArrayMemoryError:', e)
                continue
            except Exception as e:
                raise
                if len(results) == 0:
                    # error in the first dataset. Probably a new error.
                    raise
                _exc_info = sys.exc_info()
                error_message = str(e)
                print('error in data ID:', data_id, '\t', e)
                if error_message == "":
                    traceback.print_exception(*_exc_info)
                continue

            # pbar.set_description(f"{description} eyke")
            # regressor_instance = estimators[classifier_name.replace('Classifier', 'Regressor')]()
            # eyke = PairwiseDifferenceClassifierFromRegressor(estimator=regressor_instance).fit(X_train, y_train)
            # joblib.dump(eyke, f'{path}joblib/eyke_{classifier_name}_{data_id}.joblib')
            #
            # result.update(score_pairwise_difference_classifier(eyke, X_train, y_train, prefix='eyke_train_'))
            # result.update(score_pairwise_difference_classifier(eyke, X_test, y_test, prefix='eyke_test_'))
            # result.update(score_conformal_prediction(eyke, X_val, y_val, X_test, y_test.astype(int).values, prefix='eyke_'))

            # weighted = pdc
            # weighted._learn_sample_weight(X_val, y_val)
            # assert weighted.sample_weight_ is not None
            # result['weighted_test_f1'] = weighted.score(X_test, y_test)
            # result['weighted_test_f1'] = 0
            result.update(datasets.loc[data_id].to_dict())
            results.append(result)
            sys.stdout.flush()
            sys.stderr.flush()
            
            if len(results) >= DATASET_SIZE_PAPER:
                break
            else:
                if cmd_dataset_id is not None:
                    checkpoint(results=[results[-1]], path=f'{path}tmp/{get_file(classifier_name, cmd_dataset_id[i])}')
                else:
                    checkpoint(results=results, path=tmp_file)
        if len(results) == 0:
            continue
        df = pd.DataFrame(results)
        df['base_train_f1'] = df['base_train_f1_macro'].apply(np.mean).mean()
        df['pdc_train_f1'] = df['pdc_train_f1_macro'].apply(np.mean).mean()
        df['diff'] = df.pdc_test_f1 - df.base_test_f1
        # df['improvement_pdc_test_over_base'] = (df.pdc_test_f1 - df.base_test_f1) / df.base_test_f1
        # df['improvement_weighted_over_pdc'] = (df.weighted_test_f1 - df.pdc_test_f1) / df.pdc_test_f1
        columns = list(set(datasets.columns) - set(df.columns))
        datasets.index = datasets.data_id
        df[columns] = datasets.loc[df.data_id, columns].reset_index(drop=True)
        if len(df) >= 3:
            save_results(df, f'{path}/{file}')

        if len(df) > 100:
            quick_analysis(df, classifier_name, classifier_t0)

    print(f'end time {timedelta(seconds=int(time.time() - t0))}')
    # stop_mpi_workers()
