"""Performance regression gate: fails when PDC gets less accurate or much slower than the committed baseline.

Run with ``pytest -m e2e``. After an intended change of behaviour, regenerate the baseline with
``python tests/e2e/update_baseline.py`` and commit it together with the change, explaining why.
"""
import json

import numpy as np
import pytest

from benchmark import BASELINE_PATH, run, summary

pytestmark = pytest.mark.e2e

MAX_DROP_PER_CASE = 0.05   # macro-F1, one (dataset, base learner) pair
MAX_DROP_MEAN = 0.01       # macro-F1, averaged over all pairs
MAX_SLOWDOWN_SINGLE_THREAD = 3.   # DecisionTree runs single-threaded: comparable across machines
MAX_SLOWDOWN_TOTAL = 6.           # forests use every core: machine dependent, catastrophe guard only


@pytest.fixture(scope='module')
def results():
    return run(verbose=True)


@pytest.fixture(scope='module')
def baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def test_no_case_got_much_worse(results, baseline):
    drops = {f'{data}/{learner}': baseline['results'][data][learner]['pdc_f1'] - row['pdc_f1']
             for data, per_learner in results.items() for learner, row in per_learner.items()}
    worst = {case: round(drop, 4) for case, drop in drops.items() if drop > MAX_DROP_PER_CASE}
    assert not worst, f'macro-F1 dropped by more than {MAX_DROP_PER_CASE}: {worst}'


def test_mean_score_did_not_drop(results, baseline):
    now, before = summary(results)['mean_pdc_f1'], baseline['summary']['mean_pdc_f1']
    assert now >= before - MAX_DROP_MEAN, f'mean PDC macro-F1 {now:.4f} < baseline {before:.4f} - {MAX_DROP_MEAN}'


def test_pdc_still_beats_its_base_learners_on_average(results):
    stats = summary(results)
    assert stats['mean_pdc_f1'] > stats['mean_base_f1'], stats
    assert stats['pdc_wins'] > stats['pdc_losses'], stats


def test_not_much_slower(results, baseline):
    def single_thread_seconds(res):
        return float(np.sum([per_learner['DecisionTree']['pdc_seconds'] for per_learner in res.values()]))

    now, before = single_thread_seconds(results), single_thread_seconds(baseline['results'])
    assert now <= MAX_SLOWDOWN_SINGLE_THREAD * before, f'PDC(DecisionTree) took {now:.1f}s, baseline {before:.1f}s'
    total_now, total_before = summary(results)['pdc_seconds'], baseline['summary']['pdc_seconds']
    assert total_now <= MAX_SLOWDOWN_TOTAL * total_before, f'PDC took {total_now:.1f}s in total, baseline {total_before:.1f}s'
