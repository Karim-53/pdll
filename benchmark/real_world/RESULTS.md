# pdll real-world validation

131 datasets: 129 ran, 2 refused as expected (multi-output target, sparse input), 0 failed.

**classification** (91 datasets): median fit 7s, median peak memory 0.8 GB, max 2.6 GB.

**regression** (38 datasets): median fit 29s, median peak memory 1.6 GB, max 1.9 GB.

## All datasets

| dataset                                       | source                                  | task           | status    |   rows |   cols |   fit_s |   predict_s |   peak_gb |
|:----------------------------------------------|:----------------------------------------|:---------------|:----------|-------:|-------:|--------:|------------:|----------:|
| kaggle_titanic_raw                            | Kaggle (OpenML mirror)                  | classification | ok        |   1309 |      8 |    4.71 |        4.31 |      1    |
| kaggle_house_prices                           | Kaggle (OpenML mirror)                  | regression     | ok        |   1460 |     79 |  242.84 |        6.45 |      1.63 |
| kaggle_loan_prediction                        | Kaggle (OpenML mirror)                  | classification | ok        |    614 |     11 |    7.37 |        2.17 |      0.91 |
| kaggle_telco_churn                            | Kaggle (OpenML mirror)                  | classification | ok        |   7043 |     19 |   10.01 |       22.41 |      1.08 |
| kaggle_heart_failure                          | Kaggle (OpenML mirror)                  | classification | ok        |    299 |     12 |    3.08 |        0.46 |      0.41 |
| kaggle_heart_disease_comprehensive            | Kaggle (OpenML mirror)                  | classification | ok        |   1190 |     11 |    4.22 |        3.04 |      0.69 |
| kaggle_insurance_expenses                     | Kaggle (OpenML mirror)                  | regression     | ok        |   1338 |      6 |   14.18 |        2.27 |      1.58 |
| kaggle_mobile_price                           | Kaggle (OpenML mirror)                  | classification | ok        |   2000 |     20 |    7.24 |        5.66 |      0.81 |
| kaggle_students_dropout                       | Kaggle (OpenML mirror)                  | classification | ok        |   4424 |     36 |   11.48 |       16.14 |      0.88 |
| kaggle_employee_attrition                     | Kaggle (OpenML mirror)                  | classification | ok        |   4653 |      8 |    4.78 |       13.96 |      1.23 |
| kaggle_churn_telco_europa                     | Kaggle (OpenML mirror)                  | classification | ok        |   1400 |     15 |    7.13 |        3.2  |      0.65 |
| kaggle_cardiotocography                       | Kaggle (OpenML mirror)                  | classification | ok        |   2126 |     35 |    3.81 |        4.15 |      0.41 |
| kaggle_creditcard_fraud                       | Kaggle (OpenML mirror)                  | classification | ok        | 284807 |     29 |    8.37 |       24.77 |      0.59 |
| zindi_autoinland_vehicle_insurance            | Zindi (private copy, not redistributed) | classification | ok        |  12079 |     12 |    9.5  |       26.38 |      0.94 |
| sklearn_iris                                  | scikit-learn                            | classification | ok        |    150 |      4 |    0.42 |        0.28 |      0.19 |
| sklearn_wine                                  | scikit-learn                            | classification | ok        |    178 |     13 |    0.91 |        0.31 |      0.2  |
| sklearn_breast_cancer                         | scikit-learn                            | classification | ok        |    569 |     30 |    6.45 |        1.42 |      0.37 |
| sklearn_digits                                | scikit-learn                            | classification | ok        |   1797 |     64 |    9.54 |        8.88 |      0.58 |
| sklearn_olivetti_faces                        | scikit-learn                            | classification | ok        |    400 |   4096 |    2.7  |       15.86 |      0.44 |
| sklearn_diabetes                              | scikit-learn                            | regression     | ok        |    442 |     10 |   29.57 |        0.61 |      1.54 |
| sklearn_covtype                               | scikit-learn                            | classification | ok        | 581012 |     54 |   19.56 |       42.41 |      2.14 |
| sklearn_linnerud                              | scikit-learn                            | regression     | pdl_error |     20 |      3 |  nan    |      nan    |      0.16 |
| sklearn_20newsgroups_vectorized               | scikit-learn                            | classification | pdl_error |  11314 | 130107 |  nan    |      nan    |      0.27 |
| openml_dresses-sales                          | OpenML-CC18                             | classification | ok        |    500 |     12 |    7.44 |        0.67 |      0.99 |
| openml_forest_fires                           | OpenML-CTR23                            | regression     | ok        |    517 |     12 |   29.28 |        1.29 |      1.11 |
| openml_kc2                                    | OpenML-CC18                             | classification | ok        |    522 |     21 |    9.87 |        1.23 |      0.57 |
| openml_cylinder-bands                         | OpenML-CC18                             | classification | ok        |    540 |     37 |   12.29 |        0.97 |      0.75 |
| openml_climate-model-simulation-crashes       | OpenML-CC18                             | classification | ok        |    540 |     18 |    5.34 |        0.55 |      0.4  |
| openml_wdbc                                   | OpenML-CC18                             | classification | ok        |    569 |     30 |    7.01 |        0.53 |      0.41 |
| openml_ilpd                                   | OpenML-CC18                             | classification | ok        |    583 |     10 |    5.46 |        1.3  |      0.83 |
| openml_balance-scale                          | OpenML-CC18                             | classification | ok        |    625 |      4 |    5.44 |        0.81 |      0.72 |
| openml_student_performance_por                | OpenML-CTR23                            | regression     | ok        |    649 |     30 |   67.31 |        0.6  |      1.1  |
| openml_credit-approval                        | OpenML-CC18                             | classification | ok        |    690 |     15 |    4.39 |        0.71 |      0.64 |
| openml_breast-w                               | OpenML-CC18                             | classification | ok        |    699 |      9 |    2.68 |        0.41 |      0.31 |
| openml_eucalyptus                             | OpenML-CC18                             | classification | ok        |    736 |     19 |    6.33 |        0.8  |      0.68 |
| openml_blood-transfusion-service-center       | OpenML-CC18                             | classification | ok        |    748 |      4 |    3.39 |        0.81 |      0.8  |
| openml_diabetes                               | OpenML-CC18                             | classification | ok        |    768 |      8 |    4.61 |        0.94 |      0.86 |
| openml_energy_efficiency                      | OpenML-CTR23                            | regression     | ok        |    768 |      8 |   12.26 |        0.48 |      1.58 |
| openml_analcatdata_dmft                       | OpenML-CC18                             | classification | ok        |    797 |      4 |    2.58 |        0.75 |      0.61 |
| openml_cars                                   | OpenML-CTR23                            | regression     | ok        |    804 |     17 |   25.3  |        0.56 |      1.62 |
| openml_analcatdata_authorship                 | OpenML-CC18                             | classification | ok        |    841 |     70 |    9.54 |        1.2  |      0.59 |
| openml_vehicle                                | OpenML-CC18                             | classification | ok        |    846 |     18 |    6.06 |        0.91 |      0.64 |
| openml_QSAR_fish_toxicity                     | OpenML-CTR23                            | regression     | ok        |    908 |      6 |   14.03 |        0.68 |      1.48 |
| openml_tic-tac-toe                            | OpenML-CC18                             | classification | ok        |    958 |      9 |    4.13 |        0.92 |      0.8  |
| openml_vowel                                  | OpenML-CC18                             | classification | ok        |    990 |     12 |    4.68 |        0.85 |      0.46 |
| openml_credit-g                               | OpenML-CC18                             | classification | ok        |   1000 |     20 |    6.79 |        1.61 |      0.83 |
| openml_concrete_compressive_strength          | OpenML-CTR23                            | regression     | ok        |   1030 |      8 |   21.17 |        1.15 |      1.56 |
| openml_qsar-biodeg                            | OpenML-CC18                             | classification | ok        |   1055 |     41 |    9.78 |        1.43 |      0.74 |
| openml_geographical_origin_of_music           | OpenML-CTR23                            | regression     | ok        |   1059 |    116 |  384.43 |        1.71 |      1.69 |
| openml_solar_flare                            | OpenML-CTR23                            | regression     | ok        |   1066 |     10 |    9.29 |        0.47 |      0.46 |
| openml_cnae-9                                 | OpenML-CC18                             | classification | ok        |   1080 |    856 |   14.05 |       11.33 |      1.05 |
| openml_MiceProtein                            | OpenML-CC18                             | classification | ok        |   1080 |     77 |   10.97 |        1.81 |      0.68 |
| openml_pc1                                    | OpenML-CC18                             | classification | ok        |   1109 |     21 |    5.18 |        1.07 |      0.47 |
| openml_socmob                                 | OpenML-CTR23                            | regression     | ok        |   1156 |      5 |   10.22 |        0.7  |      1.39 |
| openml_Moneyball                              | OpenML-CTR23                            | regression     | ok        |   1232 |     14 |   33.5  |        1.08 |      1.56 |
| openml_banknote-authentication                | OpenML-CC18                             | classification | ok        |   1372 |      4 |    1.84 |        0.83 |      0.34 |
| openml_pc4                                    | OpenML-CC18                             | classification | ok        |   1458 |     37 |    6.58 |        1.65 |      0.55 |
| openml_cmc                                    | OpenML-CC18                             | classification | ok        |   1473 |      9 |    4.38 |        1.58 |      1.11 |
| openml_airfoil_self_noise                     | OpenML-CTR23                            | regression     | ok        |   1503 |      5 |   10.51 |        0.78 |      1.57 |
| openml_pc3                                    | OpenML-CC18                             | classification | ok        |   1563 |     37 |    6.96 |        1.53 |      0.56 |
| openml_semeion                                | OpenML-CC18                             | classification | ok        |   1593 |    256 |   14.77 |        4.86 |      1.12 |
| openml_red_wine                               | OpenML-CTR23                            | regression     | ok        |   1599 |     11 |   37.89 |        1.32 |      0.75 |
| openml_car                                    | OpenML-CC18                             | classification | ok        |   1728 |      6 |    5.5  |        2.05 |      0.5  |
| openml_steel-plates-fault                     | OpenML-CC18                             | classification | ok        |   1941 |     27 |   12.66 |        3.69 |      0.67 |
| openml_mfeat-factors                          | OpenML-CC18                             | classification | ok        |   2000 |    216 |   22.58 |        9.89 |      1.08 |
| openml_mfeat-fourier                          | OpenML-CC18                             | classification | ok        |   2000 |     76 |   21.93 |       11.06 |      0.68 |
| openml_mfeat-karhunen                         | OpenML-CC18                             | classification | ok        |   2000 |     64 |   10.96 |        2.74 |      0.66 |
| openml_mfeat-morphological                    | OpenML-CC18                             | classification | ok        |   2000 |      6 |    5.37 |        1.67 |      0.45 |
| openml_mfeat-zernike                          | OpenML-CC18                             | classification | ok        |   2000 |     47 |    9.57 |        2.85 |      0.59 |
| openml_mfeat-pixel                            | OpenML-CC18                             | classification | ok        |   2000 |    240 |   15.29 |        6.89 |      1.08 |
| openml_auction_verification                   | OpenML-CTR23                            | regression     | ok        |   2043 |      7 |   15.16 |        1.49 |      1.57 |
| openml_kc1                                    | OpenML-CC18                             | classification | ok        |   2109 |     21 |    7.02 |        2.67 |      0.58 |
| openml_segment                                | OpenML-CC18                             | classification | ok        |   2310 |     16 |    3.65 |        1.41 |      0.42 |
| openml_ozone-level-8hr                        | OpenML-CC18                             | classification | ok        |   2534 |     72 |    9.94 |        3.67 |      0.6  |
| openml_madelon                                | OpenML-CC18                             | classification | ok        |   2600 |    500 |   19.52 |       18.81 |      1.19 |
| openml_space_ga                               | OpenML-CTR23                            | regression     | ok        |   3107 |      6 |   19.41 |        2.41 |      1.57 |
| openml_dna                                    | OpenML-CC18                             | classification | ok        |   3186 |    180 |   23.13 |       11.19 |      1.1  |
| openml_splice                                 | OpenML-CC18                             | classification | ok        |   3190 |     60 |   10.78 |        5.21 |      0.77 |
| openml_kr-vs-kp                               | OpenML-CC18                             | classification | ok        |   3196 |     36 |    9.3  |        4.1  |      0.65 |
| openml_Internet-Advertisements                | OpenML-CC18                             | classification | ok        |   3279 |   1558 |   17.32 |       64.18 |      1.05 |
| openml_Bioresponse                            | OpenML-CC18                             | classification | ok        |   3751 |   1776 |    9.75 |       72.08 |      1.13 |
| openml_sick                                   | OpenML-CC18                             | classification | ok        |   3772 |     29 |    7.03 |        3.88 |      0.65 |
| openml_abalone                                | OpenML-CTR23                            | regression     | ok        |   4177 |      8 |   20.28 |        2.86 |      1.24 |
| openml_spambase                               | OpenML-CC18                             | classification | ok        |   4601 |     57 |   19.17 |        8.91 |      0.83 |
| openml_wilt                                   | OpenML-CC18                             | classification | ok        |   4839 |      5 |    3.39 |        4.07 |      0.49 |
| openml_white_wine                             | OpenML-CTR23                            | regression     | ok        |   4898 |     11 |   27.48 |        3.41 |      0.9  |
| openml_churn                                  | OpenML-CC18                             | classification | ok        |   5000 |     20 |    9.03 |        5.88 |      0.73 |
| openml_phoneme                                | OpenML-CC18                             | classification | ok        |   5404 |      5 |    3.82 |        5.84 |      0.86 |
| openml_wall-robot-navigation                  | OpenML-CC18                             | classification | ok        |   5456 |     24 |    7.21 |        5.06 |      0.75 |
| openml_texture                                | OpenML-CC18                             | classification | ok        |   5500 |     40 |    6.35 |        4.88 |      0.52 |
| openml_optdigits                              | OpenML-CC18                             | classification | ok        |   5620 |     64 |    8.91 |        8.59 |      0.62 |
| openml_first-order-theorem-proving            | OpenML-CC18                             | classification | ok        |   6118 |     51 |   14.71 |       10.86 |      0.93 |
| openml_satimage                               | OpenML-CC18                             | classification | ok        |   6430 |     36 |    7.8  |        6.17 |      0.6  |
| openml_isolet                                 | OpenML-CC18                             | classification | ok        |   7797 |    617 |   11.61 |       52.04 |      1.07 |
| openml_cpu_activity                           | OpenML-CTR23                            | regression     | ok        |   8192 |     21 |   59.23 |        6.15 |      1.35 |
| openml_kin8nm                                 | OpenML-CTR23                            | regression     | ok        |   8192 |      8 |   28.53 |        7    |      1.64 |
| openml_pumadyn32nh                            | OpenML-CTR23                            | regression     | ok        |   8192 |     32 |  108.1  |        6.29 |      1.73 |
| openml_GesturePhaseSegmentationProcessed      | OpenML-CC18                             | classification | ok        |   9873 |     32 |   11.58 |       14.07 |      0.95 |
| openml_grid_stability                         | OpenML-CTR23                            | regression     | ok        |  10000 |     12 |   28.89 |        6.69 |      1.73 |
| openml_har                                    | OpenML-CC18                             | classification | ok        |  10299 |    561 |    9.18 |       59.55 |      1.13 |
| openml_brazilian_houses                       | OpenML-CTR23                            | regression     | ok        |  10692 |      9 |   25.81 |       11.94 |      1.69 |
| openml_jm1                                    | OpenML-CC18                             | classification | ok        |  10885 |     21 |   10.13 |       14.97 |      1.01 |
| openml_pendigits                              | OpenML-CC18                             | classification | ok        |  10992 |     16 |    5.07 |        7.26 |      0.58 |
| openml_PhishingWebsites                       | OpenML-CC18                             | classification | ok        |  11055 |     30 |    7.44 |       10.76 |      0.78 |
| openml_naval_propulsion_plant                 | OpenML-CTR23                            | regression     | ok        |  11934 |     14 |   34.7  |        8.33 |      1.55 |
| openml_miami_housing                          | OpenML-CTR23                            | regression     | ok        |  13932 |     15 |   37.39 |        9.37 |      1.71 |
| openml_fifa                                   | OpenML-CTR23                            | regression     | ok        |  19178 |     28 |   73.56 |       12.84 |      1.5  |
| openml_letter                                 | OpenML-CC18                             | classification | ok        |  20000 |     16 |    4.87 |       14.95 |      0.63 |
| openml_california_housing                     | OpenML-CTR23                            | regression     | ok        |  20640 |      8 |   20.2  |       12.24 |      1.72 |
| openml_superconductivity                      | OpenML-CTR23                            | regression     | ok        |  21263 |     81 |  222.6  |       19.93 |      1.88 |
| openml_kings_county                           | OpenML-CTR23                            | regression     | ok        |  21613 |     21 |   51.2  |       13.48 |      1.72 |
| openml_health_insurance                       | OpenML-CTR23                            | regression     | ok        |  22272 |     11 |   22.09 |       10.54 |      1.46 |
| openml_fps_benchmark                          | OpenML-CTR23                            | regression     | ok        |  24624 |     43 |   95.63 |       16.53 |      1.76 |
| openml_cps88wages                             | OpenML-CTR23                            | regression     | ok        |  28155 |      6 |   10.62 |       10.83 |      1.72 |
| openml_nomao                                  | OpenML-CC18                             | classification | ok        |  34465 |    118 |   14.66 |       37.61 |      0.99 |
| openml_jungle_chess_2pcs_raw_endgame_complete | OpenML-CC18                             | classification | ok        |  44819 |      6 |    3.91 |       16.2  |      1.22 |
| openml_bank-marketing                         | OpenML-CC18                             | classification | ok        |  45211 |     16 |    6.1  |       18.68 |      0.95 |
| openml_electricity                            | OpenML-CC18                             | classification | ok        |  45312 |      8 |    5.08 |       20.29 |      1.24 |
| openml_physiochemical_protein                 | OpenML-CTR23                            | regression     | ok        |  45730 |      9 |   22.18 |       12.2  |      1.72 |
| openml_adult                                  | OpenML-CC18                             | classification | ok        |  48842 |     14 |    6.73 |       20.55 |      1.09 |
| openml_sarcos                                 | OpenML-CTR23                            | regression     | ok        |  48933 |     21 |   53.14 |       15.63 |      1.77 |
| openml_diamonds                               | OpenML-CTR23                            | regression     | ok        |  53940 |      9 |   20.14 |       11.45 |      1.73 |
| openml_CIFAR_10                               | OpenML-CC18                             | classification | ok        |  60000 |   3072 |    7.84 |      673.9  |      2.56 |
| openml_connect-4                              | OpenML-CC18                             | classification | ok        |  67557 |     42 |   12.25 |       25.38 |      1.19 |
| openml_video_transcoding                      | OpenML-CTR23                            | regression     | ok        |  68784 |     18 |   39.65 |       10.91 |      1.76 |
| openml_mnist_784                              | OpenML-CC18                             | classification | ok        |  70000 |    784 |    9.03 |      152.06 |      1.51 |
| openml_Fashion-MNIST                          | OpenML-CC18                             | classification | ok        |  70000 |    784 |    8.73 |      151.66 |      1.5  |
| openml_wave_energy                            | OpenML-CTR23                            | regression     | ok        |  72000 |     48 |  133.57 |       17.27 |      1.83 |
| openml_Devnagari-Script                       | OpenML-CC18                             | classification | ok        |  92000 |   1024 |    7.25 |      193.53 |      1.82 |
| openml_numerai28.6                            | OpenML-CC18                             | classification | ok        |  96320 |     21 |    8.7  |       23.05 |      1.14 |
| sklearn_kddcup99_SA                           | scikit-learn                            | classification | ok        | 100655 |     41 |    6.2  |       14.46 |      0.91 |
