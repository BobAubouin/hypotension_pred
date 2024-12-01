from pathlib import Path
import pickle

import optuna
import pandas as pd
import xgboost as xgb

from hp_pred.experiments import bootstrap_test, objective_xgboost

optuna.logging.set_verbosity(optuna.logging.WARNING)

SIGNAL_FEATURE = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct']
STATIC_FEATURE = ["age", "bmi", "asa"]
HALF_TIME_FILTERING = [60, 3*60, 10*60]


dataset_folder = Path("data/datasets/30_s_dataset")
model_filename = "xgb_30_s_smoteenn.json"

# import the data frame and add the meta data to the segments
data = pd.read_parquet(dataset_folder / 'cases/')

static = pd.read_parquet(dataset_folder / 'meta.parquet')

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
        f"{signal}_constant_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + [
        f"{signal}_slope_{half_time}"
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

FEATURE_NAME = [x for x in FEATURE_NAME if f"std_{HALF_TIME_FILTERING[0]}" not in x]

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
model_file = model_folder / model_filename


if model_file.exists():
    model = xgb.XGBClassifier()
    model.load_model(model_file)
else:
    # create an optuna study

    number_fold = len(train.cv_split.unique())
    data_train_cv = [train[train.cv_split != f'cv_{i}'] for i in range(number_fold)]
    data_test_cv = [train[train.cv_split == f'cv_{i}'] for i in range(number_fold)]
    # creat an optuna study
    sampler = optuna.samplers.TPESampler(seed=rng_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_xgboost(trial, data_train_cv, data_test_cv, FEATURE_NAME),
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

y_pred = model.predict_proba(test[FEATURE_NAME])
y_test = test["label"].to_numpy()
y_label_ids = test["label_id"].to_numpy()


dict_results, tprs_interpolated, precision_interpolated = bootstrap_test(
    y_test, y_pred, y_label_ids, n_bootstraps=200, rng_seed=rng_seed, strategy="targeted_recall", target=0.24)

result_folder = Path("data/results")
if not result_folder.exists():
    result_folder.exists()
roc_results = result_folder / "xgboost_roc_30_s.pkl"
with roc_results.open("wb") as f:
    pickle.dump(dict_results, f)
