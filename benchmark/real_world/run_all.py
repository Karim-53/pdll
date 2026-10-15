"""Run run_one.py on many datasets, one subprocess each, appending JSON lines to a results file.

    python run_all.py --out results/default.jsonl [--config default] [--only name1 name2] [--max-rows 20000]
                      [--timeout 1800]

Datasets already present in the output file are skipped, so the campaign can be resumed.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets  # noqa: E402


def done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)['dataset'] for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='default')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--min-rows', type=int, default=0)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--split', default='random', choices=['random', 'openml-task'])
    args = parser.parse_args()

    specs = datasets.registry()
    if args.only:
        specs = [s for s in specs if s['name'] in args.only]
    specs = [s for s in specs if s.get('rows', 0) >= args.min_rows and (args.max_rows is None or s.get('rows', 0) <= args.max_rows)]
    specs.sort(key=lambda s: s.get('rows', 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    skip = done(args.out)
    for i, spec in enumerate(specs):
        if spec['name'] in skip:
            continue
        start = time.time()
        command = [sys.executable, '-W', 'ignore', os.path.join(HERE, 'run_one.py'), spec['name'], '--config', args.config,
                   '--split', args.split]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, cwd=HERE)
            lines = [line for line in completed.stdout.splitlines() if line.startswith('{')]
            result = json.loads(lines[-1]) if lines else {
                'dataset': spec['name'], 'status': 'crash', 'error': (completed.stderr or completed.stdout)[-600:]}
        except subprocess.TimeoutExpired:
            result = {'dataset': spec['name'], 'status': 'timeout', 'error': f'more than {args.timeout}s'}
        result.setdefault('source', spec['source'])
        result.setdefault('task', spec['task'])
        result.setdefault('config', args.config)
        result['wall_seconds'] = round(time.time() - start, 1)
        with open(args.out, 'a') as f:
            f.write(json.dumps(result) + '\n')
        summary = {k: result.get(k) for k in ('status', 'shape', 'pdl', 'base', 'pdl_fit_seconds', 'pdl_predict_seconds', 'pdl_peak_gb')}
        print(f"[{i + 1}/{len(specs)}] {spec['name']}: {summary} {result.get('error', '')[:200]}", flush=True)


if __name__ == '__main__':
    main()
