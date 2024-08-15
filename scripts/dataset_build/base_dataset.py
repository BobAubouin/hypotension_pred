from hp_pred.databuilder import DataBuilder


def main():
    signal_features_names = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct']
    static_features_names = ["age", "bmi", "asa"]
    half_times = [30, 3*60, 10*60]

    databuilder = DataBuilder(
        raw_data_folder_path="./data/cases",
        signal_features_names=signal_features_names,
        static_data_path="./data/static_data.parquet",
        static_data_names=static_features_names,
        dataset_output_folder_path="./data/datasets/base_dataset",
        sampling_time=2,
        leading_time=2*60,
        prediction_window_length=8*60,
        observation_window_length=10*60,
        segment_shift=30,
        recovery_time=0,
        half_times=half_times,
    )

    databuilder.build()
    databuilder.build_meta()


if __name__ == "__main__":
    main()
