from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pyampd.ampd import find_peaks
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

SAMPLING_TIME = 0.04


def segment_cardiac_cycle(data_wav: pd.DataFrame):
    """Segments cardiac cycles by detecting troughs in the AP waveform."""

    sampling_time = data_wav['Time'].iloc[1] - data_wav['Time'].iloc[0]

    # Find the troughs (inverted peaks) of the AP waveform
    scale_ampd = int(1 / sampling_time)
    if len(data_wav['ap'].fillna(0)) == 0:
        thought = []
    else:
        thought = find_peaks(-data_wav['ap'].fillna(0), scale=scale_ampd)

    # Create an array for cycle_id assignments
    cycle_id = np.full(data_wav.shape[0], np.nan)  # Initialize with NaN

    for i in range(len(thought) - 1):
        cycle_id[thought[i] + 1: thought[i + 1] + 1] = i

    data_wav['cycle_id'] = cycle_id

    return data_wav


def mizoFeatures(y, nt=100, nh=8):

    plh1 = [0] * 8
    plh2 = [0] * 8
    n = len(y)
    t = np.linspace(0, 1, n)
    tn = np.linspace(0, 1, nt)
    sp = interp1d(t, y, kind='cubic')
    yint = sp(tn)
    v = np.diff(np.sign(np.diff(yint)))
    n_stat = int(np.sum(abs(v) > 0))
    m1 = min(n_stat, nh)
    plh1[0:m1] = np.where(abs(v) > 0)[0][0:m1]/100
    plh2[0:m1] = yint[np.where(abs(v) > 0)[0]][0:m1]
    features = [
        n_stat,
        *plh1,
        *plh2
    ]
    features = {f"mizo_{i}":  float(v) for i, v in enumerate(features)}

    return features


def extract_basic_feature_from_cycle(data_wav: pd.DataFrame):
    """Extracts basic features from each cycle in a DataFrame."""

    # Compute sample rate
    sample_rate = 1 / (data_wav['Time'].iloc[1] - data_wav['Time'].iloc[0])

    # Compute dP/dt for all rows in advance
    dPdt = data_wav['ap'].diff() * sample_rate

    # Define aggregation functions for efficient computation
    feature_dict = {
        'Time': ['last', lambda x: x.iloc[-1] - x.iloc[0]],
        'caseid': 'first',
        'ap': ['max', 'mean', 'std', 'last'],  # Max = systolic, Last = diastolic
        'dPdt': ['max', 'min', 'mean', 'std']
    }

    # Apply aggregation over each cycle_id
    features = data_wav.assign(dPdt=dPdt).groupby('cycle_id').agg(feature_dict)

    # Flatten multi-index column names
    features.columns = [
        'Time', 'cycle_duration', 'caseid',
        'cycle_systol', 'cycle_mean', 'cycle_std', 'cycle_diastol',
        'cycle_dPdt_max', 'cycle_dPdt_min', 'cycle_dPdt_mean', 'cycle_dPdt_std'
    ]

    # Compute pulse pressure
    features['cycle_pulse_pressure'] = features['cycle_systol'] - features['cycle_diastol']

    # compute the distance between consecutive cycles
    mizo_features_df = data_wav.groupby('cycle_id').apply(lambda x: mizoFeatures(x['ap'].interpolate(method='linear')))

    combined_features = pd.concat([features, mizo_features_df], axis=1)

    return combined_features.reset_index()


def validate_segment(feature_cycle: pd.DataFrame, dict_param: dict = None):

    if dict_param is None:
        dict_param = {
            'min_heart_rate': 40,
            'max_heart_rate': 150,
            'min_systol': 40,
            'max_systol': 250,
            'min_diastol': 20,
            'max_diastol': 200,
            'min_mean_pressure': 30,
            'max_mean_pressure': 200,
            'min_pulse_pressure': 10,
            'max_pulse_pressure': 150,
            'min_dPdt_max': 200,
            'max_dPdt_max': 3000,
        }

    # Validate the extracted features
    feature_cycle = feature_cycle[
        (feature_cycle['cycle_duration'] < 60/dict_param['min_heart_rate'])
        & (feature_cycle['cycle_duration'] > 60/dict_param['max_heart_rate'])
        & (feature_cycle['cycle_systol'] > dict_param['min_systol'])
        & (feature_cycle['cycle_systol'] < dict_param['max_systol'])
        & (feature_cycle['cycle_diastol'] > dict_param['min_diastol'])
        & (feature_cycle['cycle_diastol'] < dict_param['max_diastol'])
        & (feature_cycle['cycle_mean'] > dict_param['min_mean_pressure'])
        & (feature_cycle['cycle_mean'] < dict_param['max_mean_pressure'])
        & (feature_cycle['cycle_pulse_pressure'] > dict_param['min_pulse_pressure'])
        & (feature_cycle['cycle_pulse_pressure'] < dict_param['max_pulse_pressure'])
        & (feature_cycle['cycle_dPdt_max'] > dict_param['min_dPdt_max'])
        & (feature_cycle['cycle_dPdt_max'] < dict_param['max_dPdt_max'])
    ]
    # for each case id remove values that are exceeding a rolling mean of 3 values
    for caseid, case_cycle in feature_cycle.groupby('caseid'):
        for feature in case_cycle.columns:
            if feature not in ['caseid', 'Time', 'cycle_id']:
                rolling_mean = case_cycle[feature].rolling(window=10, min_periods=1).mean()
                case_cycle.mask(case_cycle[feature] > 1.2*rolling_mean, inplace=True)
                case_cycle.mask(case_cycle[feature] < 1.2*rolling_mean, inplace=True)
            case_cycle.dropna(inplace=True)

    return feature_cycle


def process_one_case(filename: str,
                     output_dir: str,
                     dict_param_verif: dict):
    data_wav = pd.read_parquet(filename, engine='pyarrow')

    data_wav.rename(columns={'SNUADC/ART': 'ap'}, inplace=True)

    # resample_data
    data_wav['Time'] = (
        data_wav.groupby('caseid', group_keys=False, sort=False)['Time']
        .transform(lambda x: x.interpolate(method='linear'))
        .round(4)
    )
    init_sampling = data_wav['Time'].iloc[1] - data_wav['Time'].iloc[0]
    data_wav = data_wav.iloc[::int(SAMPLING_TIME / init_sampling)]

    data_wav = segment_cardiac_cycle(data_wav)
    feature_cycle = extract_basic_feature_from_cycle(data_wav)
    feature_cycle = validate_segment(feature_cycle, dict_param_verif)

    # Save the extracted features
    for _, feature in feature_cycle.groupby('caseid'):
        feature.to_parquet(Path(output_dir) / f'case_{feature["caseid"].iloc[0]:04d}.parquet',
                           engine='pyarrow')


def extract_AP_waveform_features(
        output_dir: str = 'data/wav/cycle_features',
        waveform_dir: str = 'data/wav/cases/',
        group_size: int = 1,
        dict_param_verif: dict = None):

    # Load the waveform data
    print('Reading waveform data')
    file_list = list(Path(waveform_dir).glob('*.parquet'))

    # do not process if file already in the output directory
    file_list = [file for file in file_list if not (
        Path(output_dir) / f'case_{int(file.stem.split("-")[1]):04d}.parquet').exists()]

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    if len(file_list) == 0:
        raise FileNotFoundError('No waveform data found in the directory')

    process_one_case_partial = partial(process_one_case, output_dir=output_dir, dict_param_verif=dict_param_verif)

    process_map(process_one_case_partial, file_list, chunksize=1)
    print('Features extraction completed')
    return


def interpolate_patient_features(patient_signal, patient_feature):
    # Reindex features to match signal timestamps
    if patient_feature.shape[0] == 0:
        return patient_signal
    new_time = patient_signal["Time"]
    dict_feature = {'Time': new_time}
    for feature in patient_feature.columns:
        if feature != "Time":
            dict_feature[feature] = np.interp(new_time, patient_feature["Time"],
                                              patient_feature[feature].astype(float), left=np.nan, right=np.nan)

    # Create DataFrame from dict_feature
    patient_feature = pd.DataFrame(dict_feature)
    return patient_signal.merge(patient_feature, on=["caseid", "Time"], how="left")


def merge_signal_feature_cycle(
        data_signal_dir: str = 'data/cases/',
        data_feature_dir: str = 'data/wav/cycle_features/',
        output_dir: str = 'data/features/'):

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    # Load the waveform data
    print('Reading data')
    data_signal = pd.read_parquet(data_signal_dir, engine='pyarrow')
    data_feature = pd.read_parquet(data_feature_dir, engine='pyarrow')

    # keep only the intersection of the caseid
    data_signal = data_signal.query('caseid.isin(@data_feature.caseid.unique())')

    # resample data_feature at the same time as data_signal for each caseid and merge
    data_signal = data_signal.sort_values(by=['caseid', 'Time'])
    data_feature = data_feature.sort_values(by=['caseid', 'Time'])

    # Apply interpolation for each caseid
    print('Merging signal and feature data')
    merged_data = (
        data_signal.groupby("caseid", group_keys=False)
        .apply(lambda x: interpolate_patient_features(x, data_feature[data_feature["caseid"] == x["caseid"].iloc[0]]))
    )

    # Save the merged data by caseid
    print('Saving merged data')
    for caseid, data in merged_data.groupby("caseid"):
        data.to_parquet(Path(output_dir) / f'case_{caseid:04d}.parquet', engine='pyarrow')


if __name__ == '__main__':
    # print curent working directory
    # extract_AP_waveform_features()
    merge_signal_feature_cycle()
