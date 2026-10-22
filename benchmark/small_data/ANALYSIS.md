# Small-data benchmark: statistics

35 tasks x 11 models. Results imputed with the majority-class score: 0 of 385 {}

Friedman p = 4.9e-20; Nemenyi critical difference = 2.55 ranks; top group (within the CD of CatBoost): ['CatBoost', 'SVM (RBF)', 'Extra trees', 'Logistic regression', 'PDL (pdll 1.0)', 'Random forest', 'HistGradientBoosting', 'LightGBM']

| model                |   mean rank |   mean accuracy |   top-3 finishes |   median fit s |   median predict s |
|:---------------------|------------:|----------------:|-----------------:|---------------:|-------------------:|
| CatBoost             |       3.529 |           0.848 |           16.000 |        179.460 |              0.160 |
| SVM (RBF)            |       4.157 |           0.844 |           16.000 |          2.680 |              0.380 |
| Extra trees          |       5.071 |           0.839 |           13.000 |          2.290 |              0.990 |
| Logistic regression  |       5.357 |           0.831 |           13.000 |          0.600 |              0.150 |
| PDL (pdll 1.0)       |       5.486 |           0.836 |            9.000 |         65.140 |             12.460 |
| Random forest        |       5.686 |           0.838 |            9.000 |          3.130 |              0.960 |
| HistGradientBoosting |       5.814 |           0.837 |           10.000 |         13.920 |              0.550 |
| LightGBM             |       5.971 |           0.838 |            3.000 |          3.230 |              0.140 |
| XGBoost              |       6.314 |           0.837 |            6.000 |          4.910 |              0.320 |
| k-NN                 |       8.114 |           0.816 |            3.000 |          0.210 |              0.530 |
| Decision tree        |      10.500 |           0.775 |            0.000 |          0.240 |              0.090 | 

## PDL against each model

| model                |   wins |   ties |   losses |   mean_diff |   ci_low |   ci_high |   wilcoxon_p |   holm_p |   p_pdl |   p_rope |   p_other |   fit_ratio | verdict                                                              |
|:---------------------|-------:|-------:|---------:|------------:|---------:|----------:|-------------:|---------:|--------:|---------:|----------:|------------:|:---------------------------------------------------------------------|
| Logistic regression  |     14 |      7 |       14 |       0.005 |   -0.009 |     0.023 |        0.812 |    1.000 |   0.068 |    0.823 |     0.109 |     108.567 | inconclusive (95% interval of the mean difference: -0.009 to +0.023) |
| k-NN                 |     22 |      8 |        5 |       0.020 |    0.012 |     0.030 |        0.000 |    0.001 |   0.932 |    0.068 |     0.000 |     310.190 | more accurate than k-NN on these tasks                               |
| SVM (RBF)            |      6 |     13 |       16 |      -0.008 |   -0.016 |    -0.000 |        0.059 |    0.410 |   0.000 |    0.777 |     0.222 |      24.306 | inconclusive (95% interval of the mean difference: -0.016 to -0.000) |
| Decision tree        |     33 |      1 |        1 |       0.061 |    0.048 |     0.077 |        0.000 |    0.000 |   1.000 |    0.000 |     0.000 |     271.417 | more accurate than Decision tree on these tasks                      |
| Random forest        |     10 |     12 |       13 |      -0.002 |   -0.008 |     0.003 |        0.670 |    1.000 |   0.000 |    0.997 |     0.003 |      20.812 | practically equivalent to Random forest (within 1 pp)                |
| Extra trees          |      7 |     14 |       14 |      -0.002 |   -0.008 |     0.003 |        0.149 |    0.896 |   0.000 |    0.997 |     0.002 |      28.445 | practically equivalent to Extra trees (within 1 pp)                  |
| HistGradientBoosting |     12 |     12 |       11 |      -0.001 |   -0.010 |     0.008 |        0.831 |    1.000 |   0.033 |    0.917 |     0.050 |       4.680 | not practically worse than HistGradientBoosting                      |
| LightGBM             |     15 |      9 |       11 |      -0.002 |   -0.010 |     0.006 |        0.756 |    1.000 |   0.008 |    0.947 |     0.045 |      20.167 | not practically worse than LightGBM                                  |
| XGBoost              |     15 |      8 |       12 |      -0.001 |   -0.010 |     0.007 |        0.606 |    1.000 |   0.042 |    0.914 |     0.044 |      13.267 | not practically worse than XGBoost                                   |
| CatBoost             |      5 |     11 |       19 |      -0.012 |   -0.020 |    -0.005 |        0.004 |    0.035 |   0.000 |    0.596 |     0.404 |       0.363 | CatBoost is more accurate than PDL                                   | 

**Declared outcome: B (middle)** (PDL mean-rank place 5 of 11).

## Where PDL stands by size

- up to 1,000 rows (18 tasks): PDL mean rank 5.47 (place 4), mean gap to the best other model -2.4 pp
- 1,001 to 2,000 rows (17 tasks): PDL mean rank 5.50 (place 4), mean gap to the best other model -2.0 pp

## Sensitivity

- Without the 9 tasks of the defaults study (26 tasks): PDL mean rank 5.60 (place 5); against ExtraTrees: 4/10/12, Holm p 0.573, practically equivalent to Extra trees (within 1 pp). Verdicts that change: LightGBM: not practically worse than LightGBM -> inconclusive; CatBoost: CatBoost is more accurate than PDL -> inconclusive
- Also without diabetes, ilpd, pc4 (23 tasks): PDL mean rank 5.67 (place 5); against ExtraTrees: 3/10/10, Holm p 0.489, practically equivalent to Extra trees (within 1 pp). Verdicts that change: k-NN: more accurate than k-NN on these tasks -> not practically worse than k-NN; LightGBM: not practically worse than LightGBM -> inconclusive; CatBoost: CatBoost is more accurate than PDL -> inconclusive
- mfeat views averaged into one task (30 tasks): PDL mean rank 5.47 (place 5); against ExtraTrees: 6/12/12, Holm p 1.000, practically equivalent to Extra trees (within 1 pp). Verdicts that change: HistGradientBoosting: not practically worse than HistGradientBoosting -> inconclusive; XGBoost: not practically worse than XGBoost -> inconclusive; CatBoost: CatBoost is more accurate than PDL -> inconclusive
- roc_auc instead of accuracy (35 tasks): PDL mean rank 5.07 (place 3); against ExtraTrees: 14/18/3, Holm p 0.598, not practically worse than Extra trees. Verdicts that change: Logistic regression: inconclusive -> practically equivalent to Logistic regression (within 1 pp); SVM (RBF): inconclusive -> practically equivalent to SVM (RBF) (within 1 pp); Extra trees: practically equivalent to Extra trees (within 1 pp) -> not practically worse than Extra trees; HistGradientBoosting: not practically worse than HistGradientBoosting -> practically equivalent to HistGradientBoosting (within 1 pp); LightGBM: not practically worse than LightGBM -> practically equivalent to LightGBM (within 1 pp); XGBoost: not practically worse than XGBoost -> practically equivalent to XGBoost (within 1 pp); CatBoost: CatBoost is more accurate than PDL -> practically equivalent to CatBoost (within 1 pp)
- macro_f1 instead of accuracy (35 tasks): PDL mean rank 6.39 (place 9); against ExtraTrees: 7/10/18, Holm p 0.157, inconclusive (95% interval of the mean difference: -0.025 to -0.005). Verdicts that change: k-NN: more accurate than k-NN on these tasks -> not practically worse than k-NN; Random forest: practically equivalent to Random forest (within 1 pp) -> inconclusive; Extra trees: practically equivalent to Extra trees (within 1 pp) -> inconclusive; HistGradientBoosting: not practically worse than HistGradientBoosting -> inconclusive; LightGBM: not practically worse than LightGBM -> inconclusive; XGBoost: not practically worse than XGBoost -> inconclusive
- Wins/ties/losses with a tie threshold of 0 pp: {'Logistic regression': '15/1/19', 'k-NN': '29/0/6', 'SVM (RBF)': '12/1/22', 'Decision tree': '34/0/1', 'Random forest': '16/1/18', 'Extra trees': '10/4/21', 'HistGradientBoosting': '19/1/15', 'LightGBM': '19/2/14', 'XGBoost': '21/1/13', 'CatBoost': '12/1/22'}
- Wins/ties/losses with a tie threshold of 1 pp: {'Logistic regression': '9/16/10', 'k-NN': '18/13/4', 'SVM (RBF)': '5/18/12', 'Decision tree': '32/3/0', 'Random forest': '6/22/7', 'Extra trees': '7/18/10', 'HistGradientBoosting': '8/19/8', 'LightGBM': '7/19/9', 'XGBoost': '9/17/9', 'CatBoost': '3/16/16'}
