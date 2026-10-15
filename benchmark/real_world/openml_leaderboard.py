"""Run PDL on OpenML tasks with the official 10-fold splits and rank it against the public runs.

    python openml_leaderboard.py --out results/leaderboard.jsonl 31 37 1464 ...

Each run is also saved under results/openml_runs/<task id>/ so it can be uploaded to OpenML later
(``openml.runs.OpenMLRun.from_filesystem(path).publish()``, needs an OpenML API key).
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import openml
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))


def public_rank(task_id, score, metric='predictive_accuracy', page=1000, max_pages=50):
    """Distinct public flows (methods) whose best run on the task beats ``score``, and the best public score."""
    better_flows, best, offset = set(), None, 0
    for _ in range(max_pages):
        ev = openml.evaluations.list_evaluations(function=metric, tasks=[task_id], output_format='dataframe', size=page,
                                                 offset=offset, sort_order='desc')
        if ev is None or len(ev) == 0:
            break
        best = ev['value'].max() if best is None else max(best, ev['value'].max())
        better = ev[ev['value'] > score]
        better_flows.update(better['flow_name'].map(lambda name: name.split('(')[0]))  # flow family, not version/params
        if len(better) < len(ev):  # sorted descending: the rest is worse
            break
        offset += page
    return len(better_flows), best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', nargs='+', type=int)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from pdll import PairwiseDifferenceClassifier, __version__

    for task_id in args.tasks:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        model = PairwiseDifferenceClassifier(random_state=0)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False, upload_flow=False)
        seconds = time.time() - start
        run.to_filesystem(os.path.join(HERE, 'results', 'openml_runs', str(task_id)), store_model=False)
        accuracy = float(run.get_metric_fn(accuracy_score).mean())
        n_better, best = public_rank(task_id, accuracy)
        result = {'task': task_id, 'dataset': dataset.name, 'rows': int(dataset.qualities['NumberOfInstances']),
                  'pdll': __version__, 'accuracy_10fold': round(accuracy, 4), 'best_public': None if best is None else round(float(best), 4),
                  'public_methods_better': n_better, 'seconds': round(seconds, 1)}
        print(json.dumps(result), flush=True)
        with open(args.out, 'a') as f:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
