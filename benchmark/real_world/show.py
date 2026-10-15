"""Print a compact table of a results file."""
import json
import sys

for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pdl, base = r.get('pdl') or {}, r.get('base') or {}
            key = 'macro_f1' if r.get('task') == 'classification' else 'r2'
            print(f"{r['dataset'][:34]:34} {r['status']:9} {str(r.get('shape')):14} pdl {pdl.get(key, float('nan')):6.3f} "
                  f"base {base.get(key, float('nan')):6.3f} fit {r.get('pdl_fit_seconds', 0):6.1f}s "
                  f"pred {r.get('pdl_predict_seconds', 0):5.1f}s peak {r.get('pdl_peak_gb', 0):5.2f}GB  {(r.get('error') or '')[:110]}")
