#!/bin/bash

#OAR -n explain_model
#OAR -l /nodes=1/core=32,walltime=01:00:00
#OAR -stderr explain.err
#OAR -stdout explain.out
#OAR --project pr-damon

source .venv/bin/activate
export PYTHONHASHSEED=0
python3 scripts/experiments/explain_model.py
