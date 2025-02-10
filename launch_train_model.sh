#!/bin/bash

#OAR -n create_dataset
#OAR -l /nodes=1/core=32,walltime=02:00:00
#OAR -stderr dataset.err
#OAR -stdout dataset.out
#OAR --project pr-damon

conda activate hypo
export PYTHONHASHSEED=1
python3 scripts/experiments/train_model.py
# python3 scripts/experiments/explain_model.py
