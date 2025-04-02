from pathlib import Path

import numpy as np
import pandas as pd

from hp_pred.test_model import TestModel

dataset_name = '30_s_dataset'  # chu_dataset
dataset_train_name = '30_s_dataset'
# dataset_name = 'chu_dataset'
model_filename = 'xgb_no_filter.json'  # 30_s_filtered.json' xgb_cycle
model_filename_2 = 'xgb_easy_point_filtered.json'
model_filename_3 = 'xgb_inter_filtered.json'
model_filename_4 = 'xgb_inter_easy_points_filtered.json'
feature_type = "time"

data = pd.read_parquet(f'data/datasets/{dataset_name}/cases', dtype_backend='pyarrow')
if feature_type == "wave" or feature_type == "mixt":
    data_wave = pd.read_parquet(f'data/datasets/{dataset_name}/wave_rocket_features/')
    data = data.merge(data_wave, left_on=['caseid', 'time'], right_on=['caseid', 'Time'])
    data['label'] = data['label_x']

static_clean = pd.read_parquet(f'data/datasets/{dataset_name}/meta.parquet')

data = data.merge(static_clean, on='caseid')
train = pd.read_parquet(f'data/datasets/{dataset_train_name}/cases', dtype_backend='pyarrow')
static_train = pd.read_parquet(f'data/datasets/{dataset_train_name}/meta.parquet')
train = train.merge(static_train, on='caseid')

study_names = ['test_full', 'test_no_hypo', 'test_no_inter', 'test_no_hypo_no_inter']

for study_name in study_names:
    data_filter = data.copy()
    train_filter = train.copy()
    if 'no_hypo' in study_name:
        data_filter = data_filter.query('(ioh_at_time_t == 0) & (ioh_in_leading_time == 0)')
        train_filter = train_filter.query('(ioh_at_time_t == 0) & (ioh_in_leading_time == 0)')
    elif 'no_inter' in study_name:
        data_filter = data_filter.query('intervention == 0')
        train_filter = train_filter.query('intervention == 0')

    if dataset_name == dataset_train_name:
        train = data_filter.query('split == "train"')
        test = data_filter.query('split == "test"')
    else:
        test = data_filter

    tester = TestModel(
        test,
        train,
        [model_filename, model_filename_2, model_filename_3, model_filename_4],  # model_filename_2
        output_name=study_name,
        plot_name=['Model full', 'Model no_hypo', 'Model no_inter', 'Model no_hypo_ no_inter'],
        # n_bootstraps=10,
    )

    tester.test_baseline()
    tester.test_model()
    print(f"{study_name} done")
