from itertools import chain, repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats
import shap
import xgboost as xgb
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

from hp_pred.experiments import bootstrap_test, objective_xgboost, precision_event_recall, load_labelized_cases, print_statistics

optuna.logging.set_verbosity(optuna.logging.WARNING)

SIGNAL_FEATURE = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct']
STATIC_FEATURE = ["age", "bmi", "asa"]
HALF_TIME_FILTERING = [30, 3*60, 10*60]

# import the data frame and add the meta data to the segments
data = pd.read_parquet(Path('data/datasets/base_dataset/cases/'))

static = pd.read_parquet('data/datasets/base_dataset/meta.parquet')

data = data.merge(static, on='caseid')

train = data[data['split'] == "train"]
test = data[data['split'] == "test"]


# control reproducibility
rng_seed = 42


# FEATURE_NAME = (
#     [
#         f"{signal}_ema_{half_time}"
#         for signal in SIGNAL_FEATURE
#         for half_time in HALF_TIME_FILTERING
#     ]
#     + [
#         f"{signal}_std_{half_time}"
#         for signal in SIGNAL_FEATURE
#         for half_time in HALF_TIME_FILTERING
#     ]
#     + STATIC_FEATURE
# )


FEATURE_NAME = (
    [
        f"{signal}_pred_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + [
        f"{signal}_std_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + STATIC_FEATURE
)

# create a regressor
train = train.dropna(subset=FEATURE_NAME)
test = test.dropna(subset=FEATURE_NAME)
print(
    f"{len(train):,d} train samples, "
    f"{len(test):,d} test samples, "
    f"{test['label'].mean():.2%} positive rate."
)

# Set model file, create models folder if does not exist.
model_folder = Path("data/models")
if not model_folder.exists():
    model_folder.mkdir()
model_file = model_folder / "xgb.json"


if model_file.exists():
    model = xgb.XGBClassifier()
    model.load_model(model_file)
else:
    # creat an optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_xgboost(trial, train, FEATURE_NAME),
        n_trials=100,
        show_progress_bar=True,
    )

    # get the best hyperparameters
    best_params = study.best_params

    model = xgb.XGBClassifier(**best_params)
    # refit the model with best parameters
    model.fit(train[FEATURE_NAME], train.label, verbose=1)

    # save the model
    model.save_model(model_file)

y_pred = model.predict_proba(test[FEATURE_NAME])[:, 1]
y_test = test["label"].to_numpy()
y_label_ids = test["label_id"].to_numpy()

df_results, tprs_interpolated, precision_interpolated = bootstrap_test(y_test, y_pred, y_label_ids, n_bootstraps=200, rng_seed=rng_seed, strategy="targeted_recall", target=0.406)

result_folder = Path("data/results")
if not result_folder.exists():
    result_folder.mkdir()
file_results = result_folder / "xgboost_recall_fixed.csv"
df_results.to_csv(file_results, index=False)

df_results_2, tprs_interpolated, precision_interpolated = bootstrap_test(y_test, y_pred, y_label_ids, n_bootstraps=200, rng_seed=rng_seed, strategy="targeted_precision", target=0.75)

file_results_2 = result_folder / "xgboost_precision_fixed.csv"
df_results_2.to_csv(file_results_2, index=False)

