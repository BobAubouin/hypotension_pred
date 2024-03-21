import numpy as np
import pandas as pd
from pytest import TempPathFactory, fixture


from hp_pred.databuilder import DataBuilder, CASE_SUBFOLDER_NAME



@fixture(scope="session")
def data_folder(tmp_path_factory: TempPathFactory):
    _data_folder = tmp_path_factory.mktemp("cases")

    return _data_folder


@fixture(scope="session")
def raw_data():
    data = pd.DataFrame()

    n_timestamps = 200
    value = np.ones(n_timestamps) * 100
    # must be labeled as 1
    value[35:76] = 64
    # must be labeled as 1
    value[110:140] = 55
    # despite the presence of nan
    value[120:125] = np.nan
    # must be labeled as 0
    value[150:180] = np.nan

    data["Solar8000/ART_MBP"] = value
    data["Solar8000/ART_SBP"] = value
    data["Solar8000/ART_DBP"] = value

    data["caseid"] = 1
    data["Time"] = list(range(n_timestamps))

    return data


@fixture(scope="session")
def raw_data_folder(raw_data, data_folder):
    _raw_data_folder = data_folder / CASE_SUBFOLDER_NAME
    _raw_data_folder.mkdir()

    raw_data.to_parquet(_raw_data_folder / "case_0001.parquet")

    return _raw_data_folder


@fixture(scope="session")
def static_data():
    return pd.DataFrame(
        {
            "age": [77],
            "bmi": [26.3],
            "asa": [2.0],
            "preop_cr": [0.82],
            "preop_htn": [1],
            "opname": ["Low anterior resection"],
            "caseid": [1],
        }
    )


@fixture(scope="session")
def static_data_file(static_data, tmp_path_factory: TempPathFactory):
    _raw_data_folder = tmp_path_factory.mktemp("static")

    _static_data_file = _raw_data_folder / "static_data.parquet"
    static_data.to_parquet(_static_data_file)

    return _static_data_file


@fixture(scope="session")
def data_builder(data_folder, static_data_file, raw_data_folder):
    signal_features_names = [
        "mbp",
        "sbp",
        "dbp",
    ]
    static_features_names = ["age", "bmi", "asa", "preop_cr", "preop_htn"]
    half_times = [10, 60, 5 * 60]

    return DataBuilder(
        raw_data_folder_path=str(raw_data_folder),
        signal_features_names=signal_features_names,
        static_data_path=str(static_data_file),
        static_data_names=static_features_names,
        dataset_output_folder_path=data_folder / "base_dataset",
        sampling_time=2,
        leading_time=0,
        prediction_window_length=10 * 60,
        observation_window_length=5 * 60,
        segment_shift=30,
        half_times=half_times,
    )

@fixture(scope="session")
def import_raw_results(data_builder: DataBuilder):
    return data_builder._import_raw()