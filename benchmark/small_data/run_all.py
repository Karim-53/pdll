"""Run every (task, model) pair of the small-data benchmark, one process each, skipping pairs already in ``--out``.

    python run_all.py --out results/cc18_small.jsonl [--workers 3 --n-jobs 4 --timeout 1800] [--models pdl catboost ...]

Pre-registered scope (fixed before any result was looked at): the 35 OpenML-CC18 tasks with at most 2,000 rows, every
model with its default settings (see ``models.py``), the official 10-fold split of each task.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from models import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))

# OpenML-CC18 tasks with at most 2,000 instances, smallest first (from benchmark/real_world/openml_suites.csv)
CC18_SMALL = [125920, 3913, 14954, 146819, 9946, 9971, 11, 29, 15, 2079, 10101, 37, 3560, 3549, 53, 49, 3022, 31, 9957,
              9981, 146800, 3918, 10093, 3902, 23, 3903, 9964, 146821, 146817, 12, 14, 16, 18, 22, 146824]


def done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['task'], r['model']) for r in map(json.loads, f)}


def run(task, model, args):
    env = dict(os.environ, OMP_NUM_THREADS=str(args.n_jobs), OPENBLAS_NUM_THREADS=str(args.n_jobs), MKL_NUM_THREADS=str(args.n_jobs))
    command = [sys.executable, '-W', 'ignore', os.path.join(HERE, 'run_one.py'), str(task), model, '--out', args.out, '--n-jobs', str(args.n_jobs)]
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
        if (task, model) not in done(args.out):  # crashed before writing its record
            error = (completed.stderr.strip().splitlines() or ['no output'])[-1][:300]
            _append(args.out, {'task': task, 'model': model, 'status': 'crashed', 'error': error})
    except subprocess.TimeoutExpired:
        _append(args.out, {'task': task, 'model': model, 'status': 'timeout', 'error': f'more than {args.timeout} s'})
    print(f'{task} {model} done', flush=True)


def _append(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--tasks', nargs='+', type=int, default=CC18_SMALL)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)
    finished = done(args.out)
    todo = [(t, m) for t in args.tasks for m in args.models if (t, m) not in finished]
    print(f'{len(todo)} runs to do', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda pair: run(*pair, args), todo))


if __name__ == '__main__':
    main()
