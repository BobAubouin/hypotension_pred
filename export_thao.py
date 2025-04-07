from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

sampling_time = 1


def detect_ioh(window: pd.Series) -> bool:
    return (window < 65).loc[~np.isnan(window)].all()


def labelize(case_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # create the label for the case
    label_raw = (
        case_data['cycle_mean'].rolling(30, min_periods=1)
        .apply(detect_ioh)
        .fillna(0)
    )

    # Roll the window on the next self.min_time_ioh samples, see if there is a label
    label = (
        label_raw.rolling(window=30, min_periods=1)
        .max()
        .shift(-30 + 1, fill_value=0)
    )

    label_id = label.diff().clip(lower=0).cumsum().fillna(0)
    label_id = label_id.astype(int)
    label_id[label == 0] = np.nan

    return label, label_id


feature_folder = Path('data/wav/cycle_features/')
feature_output = Path('data/wav/export_wav_label/')
feature_output.mkdir()

bar = tqdm.tqdm(total=len(list(feature_folder.glob('*.parquet'))))
for filename in feature_folder.glob('*.parquet'):
    feature = pd.read_parquet(filename)
    caseid = feature.caseid.iloc[0]
    wav = pd.read_parquet(f'data/wav/cases/case-{caseid:04d}.parquet')
    wav['Time'] = wav['Time'].interpolate(method='linear').round(4)
    feature.Time = pd.to_timedelta(feature.Time, unit="s")
    wav.Time = pd.to_timedelta(wav.Time, unit="s")
    wav = wav[(wav.Time >= np.min(feature.Time)) & (wav.Time <= np.max(feature.Time))]
    feature.set_index("Time", inplace=True)
    feature = feature.resample(f"{sampling_time}s", closed='right', label='right').mean()
    label, label_id = labelize(feature)
    feature.insert(len(feature.columns), 'label', label)
    feature.insert(len(feature.columns), 'label_id', label_id)
    wav.insert(2, 'label_ioh', 0)
    wav.insert(3, 'label_prediction', 0)
    for i, data_label_id in feature.groupby('label_id'):
        time_start = np.min(data_label_id.index)
        time_end = np.max(data_label_id.index)
        wav.loc[(wav.Time >= time_start) & (wav.Time <= time_end), 'label_ioh'] = 1
        wav.loc[(wav.Time >= time_start-pd.Timedelta('10min')) &
                (wav.Time <= time_end-pd.Timedelta('2min')), 'label_prediction'] = 1
    wav.to_parquet(feature_output / f"case_{caseid:04d}.parquet")
    bar.update()
