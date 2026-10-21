"""Best public OpenML score per method on each task, for ranking our results against everything ever uploaded.

    python public_rank.py results/cc18_small.jsonl [model ...]   # writes results/openml_public.json (default: rank pdl)

Every public run on an OpenML task uses the task's official 10-fold split, so its predictive accuracy is comparable
with ours. A "method" is an OpenML flow name without its version suffix: two scikit-learn pipelines with different
steps are different methods, two hyperparameter settings of the same pipeline are the same method (its best run
counts). Only runs scoring at least the lowest accuracy among our models are fetched, which is all ranking needs.
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import openml

PAGE = 1000  # the most the OpenML API returns per call
MAX_PAGES = 200
MARGIN = 0.01  # also fetch methods slightly below our lowest score


def method(flow_name):
    return re.sub(r'\(\d+\)\s*$', '', flow_name).strip()


def best_public_per_method(task_id, floor):
    best, offset = {}, 0
    for _ in range(MAX_PAGES):
        ev = openml.evaluations.list_evaluations(function='predictive_accuracy', tasks=[task_id], output_format='dataframe',
                                                 size=PAGE, offset=offset, sort_order='desc')
        if ev is None or len(ev) == 0:
            return best, True
        for name, value in zip(ev['flow_name'], ev['value']):
            if value >= floor:
                best[method(name)] = max(best.get(method(name), 0.), float(value))
        if ev['value'].min() < floor:
            return best, True
        offset += PAGE
    return best, False  # capped: lower-ranked methods may be missing


def main():
    from report import read_records
    results_path, models = sys.argv[1], sys.argv[2:] or ['pdl']  # only the models to rank set how deep to fetch
    out = os.path.join(os.path.dirname(results_path), 'openml_public.json')
    cache = json.load(open(out)) if os.path.exists(out) else {}
    floors = {}
    for r in read_records(results_path):
        if r.get('status') == 'ok' and r.get('model') in models:
            floors[r['task']] = min(floors.get(r['task'], 1.), r['accuracy'])
    lock = threading.Lock()

    def fetch(task_id):
        floor = floors[task_id] - MARGIN
        if str(task_id) in cache and cache[str(task_id)]['floor'] <= floor:
            return
        best, complete = best_public_per_method(task_id, floor)
        with lock:
            cache[str(task_id)] = {'floor': floor, 'complete': complete, 'best_per_method': best}
            with open(out, 'w') as f:
                json.dump(cache, f, indent=0)
        print(f'{task_id}: {len(best)} public methods at or above {floor:.3f}', flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, floors))


if __name__ == '__main__':
    main()
