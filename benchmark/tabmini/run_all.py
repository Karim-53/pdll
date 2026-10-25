"""Run every (dataset, model) pair on TabMini's 44 datasets, one process each, skipping pairs already in ``--out``.

    python run_all.py --out results/tabmini.jsonl [--models pdl catboost ...] [--workers 2 --n-jobs 6 --timeout 3600]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'small_data'))  # models.py
from models import MODELS, check_environment  # noqa: E402

# TabMini's 44 binary PMLB datasets with 32-500 rows, in the order of the paper's Table 3
DATASETS = [
    'parity5', 'analcatdata_fraud', 'analcatdata_aids', 'analcatdata_bankruptcy', 'analcatdata_japansolvent', 'labor',
    'analcatdata_asbestos', 'lupus', 'postoperative_patient_data', 'analcatdata_cyyoung9302', 'analcatdata_cyyoung8092',
    'analcatdata_creditscore', 'appendicitis', 'molecular_biology_promoters', 'analcatdata_boxing1', 'mux6',
    'analcatdata_boxing2', 'hepatitis', 'corral', 'glass2', 'backache', 'prnn_crabs', 'sonar', 'biomed', 'prnn_synth',
    'analcatdata_lawsuit', 'spect', 'heart_statlog', 'breast_cancer', 'heart_h', 'hungarian', 'cleve', 'heart_c',
    'haberman', 'bupa', 'spectf', 'ionosphere', 'colic', 'horse_colic', 'house_votes_84', 'vote', 'saheart', 'clean1', 'irish',
]


def read_records(path):
    """Every complete JSON line of a results file (parallel workers append to it, so a line can be cut short)."""
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def records_of(path, name, model):
    return sum(r['dataset'] == f'tabmini:{name}' and r['model'] == model for r in read_records(path))


def run(name, model, args):
    env = dict(os.environ, OMP_NUM_THREADS=str(args.n_jobs), OPENBLAS_NUM_THREADS=str(args.n_jobs), MKL_NUM_THREADS=str(args.n_jobs))
    command = [sys.executable, '-W', 'ignore', os.path.join(HERE, 'run_one.py'), name, model, '--out', args.out, '--n-jobs', str(args.n_jobs)]
    before = records_of(args.out, name, model)
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
        if records_of(args.out, name, model) == before:  # crashed before writing its record
            error = (completed.stderr.strip().splitlines() or ['no output'])[-1][:300]
            _append(args.out, {'dataset': f'tabmini:{name}', 'model': model, 'status': 'crashed', 'error': error})
    except subprocess.TimeoutExpired:
        _append(args.out, {'dataset': f'tabmini:{name}', 'model': model, 'status': 'timeout', 'error': f'more than {args.timeout} s'})
    print(f'{name} {model} done', flush=True)


def _append(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--n-jobs', type=int, default=6)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    check_environment(args.models)
    args.out = os.path.abspath(args.out)
    done = {(r['dataset'], r['model']) for r in read_records(args.out)}
    todo = [(n, m) for n in DATASETS for m in args.models if (f'tabmini:{n}', m) not in done]
    print(f'{len(todo)} runs to do', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: run(*job, args), todo))


if __name__ == '__main__':
    main()
