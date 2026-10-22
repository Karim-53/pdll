# pdll 0.3.1 against 1.0 on raw OpenML data

Both versions, with default settings, on the 21 OpenML-CC18 tasks with at most 1,100 rows: the official 10-fold
split of each task, mean accuracy over the folds. Run on 2026-10-21.

- **pdll 1.0** gets each fold exactly as OpenML returns it: a pandas DataFrame (categorical columns, missing values,
  a non-default index) and a categorical target. Its model: `PairwiseDifferenceClassifier(random_state=0)`.
- **pdll 0.3.1, raw DataFrame**: the same input, the way a user would have passed it.
- **pdll 0.3.1, pre-encoded by hand**: 0.3.1's best case, which sidesteps its input bugs: numpy arrays, categorical
  columns ordinal-encoded (NaN kept), labels encoded as 0..K-1. This variant was added after a first run showed
  that 0.3.1's raw output had the wrong labels.
- Both versions use the same base learner, `ExtraTreesClassifier(class_weight="balanced")`. 0.3.1 trains on every
  pair and uses every training row as an anchor; 1.0 caps them at 100,000 pairs and 500 anchors.
- **These results come from the first version of `run_task.py` (commit `2781b30`).** In that version 0.3.1's base
  learner was unseeded (its constructor has no `random_state`) and the best-case encoding read the whole dataset.
  None of these 21 tasks has text columns (their categories come from the OpenML dataset declaration, which 1.0
  uses too), so the encoding difference does not matter here; the missing seed means a rerun of 0.3.1 gives
  slightly different numbers, so read differences of up to ±0.005 as ties. The current script seeds 0.3.1 and
  encodes text columns per fold; it was not rerun.
- Each run has its own process, a timeout (45 min, 60 min for the last 0.3.1 run) and a 10 GB memory watchdog.
- The rank in brackets is among the public OpenML methods on the same task and split: 1 + the number of methods
  whose best public run beats the score by more than 0.0005 (a method = an OpenML flow name without its version
  suffix). "≥" marks two tasks where the download of public runs hit its page cap. Ranks below the lowest downloaded
  public run are left out (the near-chance 0.3.1 raw scores).

## Summary

- **1.0 ran on all 21 tasks** from raw DataFrames.
- **0.3.1 on the same raw input**:
  - 14 errors: category and unsigned-integer columns were rejected;
  - 4 silently wrong results, 0.03 to 0.14 accuracy where its best case scores 0.81 to 0.97. Two bugs combine: a
    fold's pandas index is not 0..n-1 and was aligned by label, and categorical labels came back in the wrong order;
  - 1 correct but lower result (diabetes 0.708 against 0.762 for its best case);
  - 2 without a result: out of memory on cnae-9 (945,000 pairs × 2,568 features) and no result within 60 minutes on
    MiceProtein.
- **Accuracy against 0.3.1's best case** (19 tasks where it finished): 1.0 is better on 3, within ±0.005 on 14 and
  lower on 2; the mean difference is +0.0007.
  - The two losses are tic-tac-toe (−0.058) and vowel (−0.015). They come from 1.0's default caps on pairs and
    anchors. `PairwiseDifferenceClassifier(max_pairs=None, max_anchors=None)` trains on every pair and anchor, as
    0.3.1 did.
- The paper's experiments passed numpy arrays and are not affected by these input bugs; `tests/test_golden.py`
  checks that 1.0 reproduces 0.3.1's outputs on them.

## Results

| Dataset (OpenML task) | Rows | Metric | pdll 0.3.1, raw DataFrame | pdll 0.3.1, pre-encoded by hand | pdll 1.0 | Best public score |
|---|---|---|---|---|---|---|
| [dresses-sales](https://www.openml.org/t/125920) | 500 | accuracy | error | 0.578 (97th) | 0.582 (85th) | 0.658 |
| [kc2](https://www.openml.org/t/3913) | 522 | accuracy | error | 0.824 (243rd) | 0.828 (220th) | 0.870 |
| [climate-model-simulation-crashes](https://www.openml.org/t/146819) | 540 | accuracy | 0.080 wrong labels | 0.918 (40th) | 0.915 (43rd) | 0.967 |
| [cylinder-bands](https://www.openml.org/t/14954) | 540 | accuracy | error | 0.832 (8th) | 0.832 (8th) | 0.859 |
| [wdbc](https://www.openml.org/t/9946) | 569 | accuracy | 0.033 wrong labels | 0.972 (44th) | 0.972 (44th) | 0.984 |
| [ilpd](https://www.openml.org/t/9971) | 583 | accuracy | error | 0.727 (21st) | 0.727 (21st) | 0.743 |
| [balance-scale](https://www.openml.org/t/11) | 625 | accuracy | 0.136 wrong labels | 0.810 (363rd) | 0.856 (279th) | 1.000 |
| [credit-approval](https://www.openml.org/t/29) | 690 | accuracy | error | 0.870 (41st) | 0.868 (51st) | 0.888 |
| [breast-w](https://www.openml.org/t/15) | 699 | accuracy | 0.135 wrong labels | 0.969 (84th) | 0.971 (26th) | 0.977 |
| [eucalyptus](https://www.openml.org/t/2079) | 736 | accuracy | error | 0.672 (40th) | 0.675 (29th) | 0.700 |
| [blood-transfusion-service-center](https://www.openml.org/t/10101) | 748 | accuracy | error | 0.739 (≥ 249th) | 0.761 (≥ 249th) | 0.807 |
| [diabetes](https://www.openml.org/t/37) | 768 | accuracy | 0.708 | 0.762 (240th) | 0.772 (105th) | 0.788 |
| [analcatdata_dmft](https://www.openml.org/t/3560) | 797 | accuracy | error | 0.195 (228th) | 0.198 (205th) | 0.263 |
| [analcatdata_authorship](https://www.openml.org/t/3549) | 841 | accuracy | error | 0.995 (52nd) | 0.994 (81st) | 0.999 |
| [vehicle](https://www.openml.org/t/53) | 846 | accuracy | error | 0.734 (167th) | 0.735 (166th) | 0.870 |
| [tic-tac-toe](https://www.openml.org/t/49) | 958 | accuracy | error | 0.965 (236th) | 0.907 (303rd) | 1.000 |
| [vowel](https://www.openml.org/t/3022) | 990 | accuracy | error | 0.991 (20th) | 0.976 (43rd) | 0.998 |
| [credit-g](https://www.openml.org/t/31) | 1000 | accuracy | error | 0.766 (≥ 54th) | 0.765 (≥ 58th) | 0.786 |
| [qsar-biodeg](https://www.openml.org/t/9957) | 1055 | accuracy | error | 0.872 (41st) | 0.869 (59th) | 0.894 |
| [MiceProtein](https://www.openml.org/t/146800) | 1080 | accuracy | no result in 60 min | no result in 60 min | 0.997 (15th) | 1.000 |
| [cnae-9](https://www.openml.org/t/9981) | 1080 | accuracy | out of memory | out of memory | 0.919 (137th) | 0.966 |

## Reproduce

```shell
pip install -e ".[test]" openml psutil
pip download pdll==0.3.1 --no-deps -d wheel && python -c "import zipfile, glob; zipfile.ZipFile(glob.glob('wheel/*.whl')[0]).extractall('pdll031')"
bash driver.sh                     # pdll 1.0 then 0.3.1 on the 21 tasks, a few hours
python make_table.py results/v031_vs_v1.jsonl ../small_data/results/openml_public.json > results/table.md
```

`results/v031_vs_v1.jsonl` holds every record; when a task was run twice, the latest record counts.
