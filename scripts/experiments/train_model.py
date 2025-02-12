from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb

from hp_pred.experiments import bootstrap_test, objective_xgboost

optuna.logging.set_verbosity(optuna.logging.WARNING)

SIGNAL_FEATURE = (['hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp']
                   + ['mbp', 'dbp', 'sbp']
                #  + ['cycle_mean', 'cycle_systol', 'cycle_diastol']
                #  + ['cycle_std', 'cycle_pulse_pressure']
                #   ['cycle_dPdt_max', 'cycle_dPdt_min', 'cycle_dPdt_mean', 'cycle_dPdt_std']
                )
STATIC_FEATURE = ["age", "bmi", "asa"]
HALF_TIME_FILTERING = [60, 3*60, 10*60]


dataset_folder = Path("data/datasets/30_s_dataset")
model_filename = "xgb_not_filter.json"
feature_type = "time"

# import the data frame and add the meta data to the segments
# dataset_folder_bis = Path("data/datasets/30_s_filtered_v2_dataset")
# other_static = pd.read_parquet(dataset_folder_bis / 'meta.parquet')

data = pd.read_parquet(dataset_folder / 'cases/')

if feature_type == "wave" or feature_type == "mixt":
    data_wave = pd.read_parquet(dataset_folder / "wave_rocket_features/")
    data = data.merge(data_wave, left_on=['caseid', 'time'], right_on=['caseid', 'Time'])
    data['label'] = data['label_x']

static = pd.read_parquet(dataset_folder / 'meta.parquet')

data = data.merge(static, on='caseid')
# data = data[data['intervention']==0]

train = data[~(data['split']=='train')]
test = data[data['split']=='test']


# control reproducibility
rng_seed = 42


if feature_type == "wave":
    FEATURE_NAME = [col for col in data.columns if "feature" in col]

elif feature_type == "time":
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
elif feature_type == "mixt":
    FEATURE_NAME = [col for col in data.columns if "feature" in col]
    FEATURE_NAME += (
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
