#!/bin/bash

#OAR -n create_dataset
#OAR -l /nodes=1/core=32,walltime=03:00:00
#OAR -stderr dataset.err
#OAR -stdout dataset.out
#OAR --project pr-damon

conda activate hypo
export PYTHONHASHSEED=0
# python3 -m hp_pred.dataset_download -s 100
# python3 src/hp_pred/extract_AP_waveform_features.py
python3 scripts/dataset_build/30_s_waveform_dataset.py
python3 scripts/experiments/train_model.py
# python3 scripts/experiments/explain_model.py