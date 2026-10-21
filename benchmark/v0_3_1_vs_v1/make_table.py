"""pdll 0.3.1 vs 1.0 on the 21 OpenML-CC18 tasks with at most 1,100 rows: 10-fold accuracy and rank among public methods.

    python make_table.py results/v031_vs_v1.jsonl ../small_data/results/openml_public.json > results/table.md
"""
import json
import sys

PUBLIC_TIE = 0.0005


def ordinal(n):
    return f'{n}{"th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'


def rank(public, task, score):
    entry = public.get(str(task))
    if entry is None or score is None or score < entry['floor']:  # below the downloaded public runs: rank unknown
        return ''
    better = sum(v > score + PUBLIC_TIE for v in entry['best_per_method'].values())
    return ('' if entry['complete'] else '≥ ') + ordinal(better + 1)


def cell(score, public, task, note=''):
    r = rank(public, task, score)
    return f'{score:.3f} ({r}){note}' if r else f'{score:.3f}{note}'


def main():
    rows = {}
    for line in open(sys.argv[1]):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[(r['task'], r['pdll'])] = r  # the latest record wins
    public = json.load(open(sys.argv[2]))
    print('| Dataset (OpenML task) | Rows | Metric | pdll 0.3.1, raw DataFrame | pdll 0.3.1, pre-encoded by hand | pdll 1.0 | Best public score |')
    print('|---|---|---|---|---|---|---|')
    tasks = sorted({t for t, v in rows if v == '1.0.0'}, key=lambda t: rows[(t, '1.0.0')]['rows'])
    for task in tasks:
        now, old = rows[(task, '1.0.0')], rows.get((task, '0.3.1'), {})
        if old.get('status') == 'out_of_memory':
            raw = encoded = 'out of memory'
        elif old.get('status') == 'timeout':
            raw = encoded = f"no result in {old['seconds'] // 60} min"
        elif not old:
            raw = encoded = 'not run'
        else:
            raw = 'error' if old.get('raw_error') else cell(old['raw_accuracy_10fold'], public, task)
            if not old.get('raw_error') and old['raw_accuracy_10fold'] < 0.5 * (old.get('encoded_accuracy_10fold') or 1):
                raw += ' wrong labels'
            encoded = 'error' if old.get('encoded_error') else cell(old['encoded_accuracy_10fold'], public, task)
        best = max(public[str(task)]['best_per_method'].values()) if str(task) in public else None
        name = f"[{now['dataset']}](https://www.openml.org/t/{task})"
        print(f"| {name} | {now['rows']} | accuracy | {raw} | {encoded} | {cell(now['accuracy_10fold'], public, task)} | "
              f"{'' if best is None else f'{best:.3f}'} |")


if __name__ == '__main__':
    main()
