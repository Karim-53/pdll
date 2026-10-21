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

from models import MODELS, check_environment

HERE = os.path.dirname(os.path.abspath(__file__))

# OpenML-CC18 tasks with at most 2,000 instances, smallest first (from benchmark/real_world/openml_suites.csv)
CC18_SMALL = [125920, 3913, 14954, 146819, 9946, 9971, 11, 29, 15, 2079, 10101, 37, 3560, 3549, 53, 49, 3022, 31, 9957,
              9981, 146800, 3918, 10093, 3902, 23, 3903, 9964, 146821, 146817, 12, 14, 16, 18, 22, 146824]


def latest_status(path):
    """{(task, model): status of its latest record}; lines cut short by concurrent appends are skipped."""
    from report import read_records
    return {(r['task'], r['model']): r['status'] for r in read_records(path)} if os.path.exists(path) else {}


def records_of(path, task, model):
    from report import read_records
    return sum(r['task'] == task and r['model'] == model for r in read_records(path)) if os.path.exists(path) else 0


def run(task, model, args):
    env = dict(os.environ, OMP_NUM_THREADS=str(args.n_jobs), OPENBLAS_NUM_THREADS=str(args.n_jobs), MKL_NUM_THREADS=str(args.n_jobs))
    command = [sys.executable, '-W', 'ignore', os.path.join(HERE, 'run_one.py'), str(task), model, '--out', args.out, '--n-jobs', str(args.n_jobs)]
    before = records_of(args.out, task, model)
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
        if records_of(args.out, task, model) == before:  # crashed before writing its record
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
    parser.add_argument('--retry-failed', action='store_true',
                        help='also rerun pairs whose latest record is an error, crash or timeout (e.g. after running out of memory)')
    args = parser.parse_args()
    check_environment(args.models)
    args.out = os.path.abspath(args.out)
    status = latest_status(args.out)
    finished = {pair for pair, s in status.items() if s == 'ok' or not args.retry_failed}
    todo = [(t, m) for t in args.tasks for m in args.models if (t, m) not in finished]
    print(f'{len(todo)} runs to do', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda pair: run(*pair, args), todo))


if __name__ == '__main__':
    main()
