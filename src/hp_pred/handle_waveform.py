from pathlib import Path

import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sktime.transformations.panel.rocket import MiniRocket


NEW_SAMPLE_TIME = "20ms"


def extract_feature_from_dir(dataset_dir: str,
                             segments_length: int,
                             extraction_method: str,
                             output_dir_name: str,
                             extraction_parameters: dict = {},
                             cases_id_list: list = None,
                             batch_size: int = 100):

    segments_length = pd.to_timedelta(segments_length, unit='s')
    data_case_dir = Path(dataset_dir) / "cases"
    data_waveform_dir = Path('data/wav/cases')
    output_dir = Path(dataset_dir) / output_dir_name
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    # get segments data
    segments_data = pd.read_parquet(Path(data_case_dir),
                                    columns=['label', 'time', 'caseid'])

    if cases_id_list is None:
        cases_id_list = segments_data.caseid.unique()

    segments_data = segments_data[segments_data.caseid.isin(cases_id_list)]
    segments_data['time'] = pd.to_timedelta(segments_data['time'], unit='s')

    batch_number = len(cases_id_list) // batch_size
    batch_number += 1 if len(cases_id_list) % batch_size != 0 else 0

    # separate the cases_id list in batch
    cases_id_list_batch = [cases_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_number-1)]
    cases_id_list_batch.append(cases_id_list[batch_size * (batch_number-1):])

    if extraction_method == 'rocket':
        rocket_model = None
    for batch in range(batch_number):
        print(f"Starting batch {batch}/{batch_number}")

        cases_id_batch = cases_id_list_batch[batch]
        segments_data_batch = segments_data[segments_data.caseid.isin(cases_id_batch)]

        data_wave_batch = []

        for case_id in cases_id_batch:
            filename = Path(data_waveform_dir) / f"case-{case_id:04d}.parquet"
            if filename.exists():
                data_wave_batch.append(
                    pd.read_parquet(filename)
                )

        data_wave_batch = pd.concat(data_wave_batch)
        data_wave_batch.rename(columns={'SNUADC/ART': 'bp'}, inplace=True)

        # Handle time data and resampling
        data_wave_batch = resample_waveform(data_wave_batch)

        # fill missing data
        data_wave_batch.ffill(inplace=True)
        data_wave_batch.bfill(inplace=True)

        # create segments
        segmented_wave = create_segments_for_batch(
            segments_data_batch,
            data_wave_batch,
            segments_length
        )

        if extraction_method == 'ts_fresh':
            if 'ts_fresh_method' not in extraction_parameters:
                extraction_parameters['ts_fresh_method'] = 'minimal'
            features = extract_feature_with_ts_fresh(segmented_wave,
                                                     extraction_parameters,
                                                     disable_progressbar=False)
        elif extraction_method == 'rocket':
            features, rocket_model = extract_feature_with_rocket(segmented_wave,
                                                                 extraction_parameters,
                                                                 rocket_model)
        else:
            raise ValueError(f"Extraction method {extraction_method} is not supported.")

        for case_id in cases_id_batch:
            features_case = features[features.id.str.contains(f"{case_id}_")]
            features_case.to_parquet(output_dir / f"case_{case_id:04d}.parquet")

    return


def resample_waveform(data_wave: pd.DataFrame):
    data_wave['Time'] = data_wave['Time'].interpolate(method='linear').round(4)
    data_wave.Time = pd.to_timedelta(data_wave.Time, unit="s")
    df_list = []
    for _, df_case in data_wave.groupby('caseid'):
        df_case.set_index('Time', inplace=True)
        df_resampled = df_case.resample(NEW_SAMPLE_TIME, closed='right', label='right').last()
        df_resampled['Time'] = df_resampled.index
        df_resampled.reset_index(drop=True, inplace=True)
        df_list.append(df_resampled)
    data_wave = pd.concat(df_list, ignore_index=True)
    return data_wave


def create_segments_for_one_case(segments_timming: pd.DataFrame,
                                 data_wave: pd.DataFrame,
                                 segments_length: pd.Timedelta):
    segmented_wave = []
    data_wave.set_index('Time', inplace=True)
    for i, row in segments_timming.iterrows():
        segment = data_wave.loc[row['time'] - segments_length:row['time']].copy()
        segment['label'] = row['label']
        segment['Time'] = segment.index
        segment['id'] = f"{row['caseid']}_{i}"  # create id
        # drop time index
        segment.reset_index(drop=True, inplace=True)
        segmented_wave.append(segment)

    segmented_wave = pd.concat(segmented_wave)
    return segmented_wave


def create_segments_for_batch(segments_timming: pd.DataFrame,
                              data_wave: pd.DataFrame,
                              segments_length: pd.Timedelta):
    segmented_wave = []
    for caseid, segments_timming_case in segments_timming.groupby('caseid'):
        data_wave_case = data_wave[data_wave['caseid'] == caseid]
        segmented_wave_case = create_segments_for_one_case(
            segments_timming_case,
            data_wave_case,
            segments_length
        )
        segmented_wave.append(segmented_wave_case)
    segmented_wave = pd.concat(segmented_wave)
    return segmented_wave


def extract_feature_with_ts_fresh(segmented_wave: pd.DataFrame,
                                  extraction_parameters: dict,
                                  disable_progressbar: bool = False):
    if extraction_parameters['ts_fresh_method'] == 'minimal':
        extraction_parameters = MinimalFCParameters()
    elif extraction_parameters['ts_fresh_method'] == 'efficient':
        extraction_parameters = EfficientFCParameters()

    features = extract_features(segmented_wave,
                                column_id='id',
                                column_sort='Time',
                                column_value='bp',
                                default_fc_parameters=extraction_parameters,
                                disable_progressbar=disable_progressbar,
                                )
    features['id'] = features.index

    # keep only the row with the biggest time of each id id segmented wave
    segmented_wave = segmented_wave.loc[segmented_wave.groupby('id')['Time'].idxmax()]
    segmented_wave.drop_duplicates(subset=['id'], inplace=True)
    features = features.merge(segmented_wave[['id', 'label', 'Time', 'caseid']], on='id', how='outer')
    return features


def extract_feature_with_rocket(segmented_wave: pd.DataFrame,
                                extraction_parameters: dict,
                                rocket_model=None):
    """
    Extract features using Rocket transformation.
    """
    # Ensure fixed random state for reproducibility
    random_state = extraction_parameters.get('random_state', 42)
    num_kernels = extraction_parameters.get('num_kernels', 1000)

    # Rocket requires data to be in a 3D array: [n_samples, n_timestamps, n_features]
    grouped_segments = []
    ids = []

    for idx, segment in segmented_wave.groupby("id"):
        ids.append(idx)
        grouped_segments.append([segment['bp'].values])

    # Convert to numpy array [n_samples, n_timestamps]
    segmented_array = np.array(grouped_segments)

    if rocket_model is not None:
        mini_rocket = rocket_model
    else:
        # Initialize Rocket with reproducible random state
        mini_rocket = MiniRocket(num_kernels=num_kernels, random_state=random_state, n_jobs=-1)
        # Fit and transform
        mini_rocket.fit(segmented_array)

    features = mini_rocket.transform(segmented_array)

    # Convert features to DataFrame and add IDs
    features.columns = [f"feature_{i}" for i in range(features.shape[1])]
    features['id'] = ids
    # keep only the row with the biggest time of each id id segmented wave
    segmented_wave = segmented_wave.loc[segmented_wave.groupby('id')['Time'].idxmax()]
    segmented_wave.drop_duplicates(subset=['id'], inplace=True)
    feature_df = features.merge(segmented_wave[['id', 'label', 'Time', 'caseid']], on='id', how='outer')

    return feature_df, rocket_model


def select_fetures(data: pd.DataFrame):
    just_features = data[[col for col in data.columns if col not in ['id', 'label', 'time', 'caseid']]]
    just_features = impute(just_features)
    usefull_features = select_features(just_features, data['label'])
    data = pd.merge(data[['id', 'label', 'time', 'caseid']], usefull_features, on='id')
    return data
