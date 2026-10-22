"""Download the best public OpenML score per method for the 21 tasks, deep enough to rank both versions.

    python fetch_public.py results/v031_vs_v1.jsonl     # writes results/openml_public.json

Uses ../small_data/public_rank.py: a method is an OpenML flow name without its version suffix, and only runs at or
above the lowest score to rank (1.0 or 0.3.1's best case, minus 0.01) are fetched.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'small_data'))
from public_rank import MARGIN, best_public_per_method  # noqa: E402


def main():
    results = sys.argv[1]
    out = os.path.join(os.path.dirname(results), 'openml_public.json')
    floors = {}
    for line in open(results):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ('accuracy_10fold', 'encoded_accuracy_10fold'):
            if r.get(key) is not None:
                floors[r['task']] = min(floors.get(r['task'], 1.), r[key])
    cache = json.load(open(out)) if os.path.exists(out) else {}
    for task, floor in sorted(floors.items()):
        if str(task) in cache and cache[str(task)]['floor'] <= floor - MARGIN:
            continue
        best, complete = best_public_per_method(task, floor - MARGIN)
        cache[str(task)] = {'floor': floor - MARGIN, 'complete': complete, 'best_per_method': best}
        with open(out, 'w') as f:
            json.dump(cache, f, indent=0)
        print(f'{task}: {len(best)} public methods at or above {floor - MARGIN:.3f}', flush=True)


if __name__ == '__main__':
    main()
