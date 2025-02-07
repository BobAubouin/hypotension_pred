from pathlib import Path

import pandas as pd
import numpy as np
from pyampd.ampd import find_peaks
from tqdm import tqdm

SAMPLING_TIME = 0.04


def segment_cardiac_cycle(data_wav: pd.DataFrame):
    """Segments cardiac cycles by detecting troughs in the AP waveform."""

    sampling_time = data_wav['Time'].iloc[1] - data_wav['Time'].iloc[0]

    # Find the troughs (inverted peaks) of the AP waveform
    scale_ampd = int(0.5 / sampling_time)
    thought = find_peaks(-data_wav['ap'].fillna(0), scale=scale_ampd)

    # Create an array for cycle_id assignments
    cycle_id = np.full(data_wav.shape[0], np.nan)  # Initialize with NaN

    for i in range(len(thought) - 1):
        cycle_id[thought[i] + 1: thought[i + 1] + 1] = i

    data_wav['cycle_id'] = cycle_id

    return data_wav


def extract_basic_feature_from_cycle(data_wav: pd.DataFrame):
    """Extracts basic features from each cycle in a DataFrame."""

    # Compute sample rate
    sample_rate = 1 / (data_wav['Time'].iloc[1] - data_wav['Time'].iloc[0])

    # Compute dP/dt for all rows in advance
    dPdt = data_wav['ap'].diff().rolling(3).mean() * sample_rate

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

    return features.reset_index()


def validate_segment(feature_cycle: pd.DataFrame, dict_param: dict = None):

    if dict_param is None:
        dict_param = {
            'min_heart_rate': 25,
            'max_heart_rate': 230,
            'min_systol': 40,
            'max_systol': 250,
            'min_diastol': 20,
            'max_diastol': 200,
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
        & (feature_cycle['cycle_pulse_pressure'] > dict_param['min_pulse_pressure'])
        & (feature_cycle['cycle_pulse_pressure'] < dict_param['max_pulse_pressure'])
        & (feature_cycle['cycle_dPdt_max'] > dict_param['min_dPdt_max'])
        & (feature_cycle['cycle_dPdt_max'] < dict_param['max_dPdt_max'])
    ]

    return feature_cycle


def extract_AP_waveform_features(
        output_dir: str,
        waveform_dir: str = 'data/wav/cases/',
        group_size: int = 5,
        dict_param_verif: dict = None):

    # Load the waveform data
    print('Reading waveform data')
    file_list = list(Path(waveform_dir).glob('*.parquet'))

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    if len(file_list) == 0:
        raise FileNotFoundError('No waveform data found in the directory')

    # Read the first file to get the column names
    iter_number = len(file_list) // group_size + (1 if len(file_list) % group_size > 0 else 0)

    for i in tqdm(range(iter_number)):
        file_group = file_list[i * group_size: (i + 1) * group_size]
        data_wav = pd.concat([pd.read_parquet(file, engine='pyarrow') for file in file_group])

        data_wav.rename(columns={'SNUADC/ART': 'ap'}, inplace=True)

        # resample_data
        data_wav['Time'] = (
            data_wav.groupby('caseid', group_keys=False, sort=False)['Time']
            .transform(lambda x: x.interpolate(method='linear'))
            .round(4)
        )

        # Get min/max times for each caseid
        min_times = data_wav.groupby('caseid')['Time'].min()
        max_times = data_wav.groupby('caseid')['Time'].max()

        new_data = []
        for case in min_times.index:
            new_times = np.arange(min_times[case], max_times[case] + SAMPLING_TIME, SAMPLING_TIME)
            new_data.append(pd.DataFrame({'caseid': case, 'Time': new_times}))

        # Combine all caseid groups back together
        new_data = pd.concat(new_data, ignore_index=True)
        data_wav = new_data.merge(data_wav, on=['caseid', 'Time'], how='left').ffill()

        data_wav = segment_cardiac_cycle(data_wav)
        feature_cycle = extract_basic_feature_from_cycle(data_wav)
        feature_cycle = validate_segment(feature_cycle, dict_param_verif)

        # Save the extracted features
        for _, feature in feature_cycle.groupby('caseid'):
            feature.to_parquet(Path(output_dir) / f'case{feature["caseid"].iloc[0]:04d}.parquet',
                               engine='pyarrow')
    print('Features extraction completed')
    return


def interpolate_patient_features(patient_signal, patient_feature):
    # Reindex features to match signal timestamps
    patient_feature = patient_feature.set_index("Time")
    patient_signal = patient_signal.set_index("Time")

    # Reindex and interpolate
    patient_feature = patient_feature.reindex(patient_signal.index).astype(float).interpolate(method='index')

    # Reset index and return merged DataFrame
    return patient_signal.reset_index().merge(patient_feature.reset_index(), on=["caseid", "Time"], how="left")


def merge_signal_feature_cycle(
        data_signal_dir: str = 'data/cases/',
        data_feature_dir: str = 'data/wav/features/',
        output_dir: str = 'data/feature/'):

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    # Load the waveform data
    print('Reading data')
    data_signal = pd.read_parquet(data_signal_dir, engine='pyarrow')
    data_feature = pd.read_parquet(data_feature_dir, engine='pyarrow')

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
    # extract_AP_waveform_features(output_dir='data/wav/features')
    merge_signal_feature_cycle()
