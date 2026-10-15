#!/usr/bin/env bash
# Defaults study (2026-10-15): which max_pairs / max_anchors / leaf size should pdll 1.0 ship with?
# Mid-size datasets, where the defaults actually bind (more than 250k training pairs).
cd "$(dirname "$0")"
PY=../../.venv/Scripts/python
DATA="openml_credit-g openml_vehicle openml_qsar-biodeg openml_pc1 openml_banknote-authentication openml_cmc openml_car
      openml_steel-plates-fault openml_mfeat-morphological openml_segment openml_kr-vs-kp openml_spambase openml_phoneme
      kaggle_titanic_raw kaggle_mobile_price openml_concrete_compressive_strength openml_airfoil_self_noise openml_red_wine
      openml_abalone kaggle_house_prices"
CLASSIFICATION=$(echo $DATA | tr ' ' '
' | grep -vE "concrete|airfoil|red_wine|abalone|house_prices" | tr '
' ' ')
CONFIGS="pairs250k_anchors500_leaf5 pairs250k_anchors500 pairs250k_anchors500_leaf3 pairs250k_anchors500_leaf10
         pairs250k_anchors1000_leaf5 pairs100k_anchors500"
for config in $CONFIGS; do
  $PY run_all.py --out results/study_$config.jsonl --config $config --timeout 1500 --only $DATA
done
# the 0.3.1/1.0.0 default, classification only (regression is already known to time out, see results/pilot_v1.0.0.jsonl)
$PY run_all.py --out results/study_pairs1M_anchors1000.jsonl --config pairs1M_anchors1000 --timeout 1500 --only $CLASSIFICATION
