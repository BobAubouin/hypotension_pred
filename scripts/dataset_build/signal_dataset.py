from hp_pred.databuilder import DataBuilder


def main():
    signal_features_names = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct']
    static_features_names = ["age", "bmi", "asa", "preop_cr", "preop_htn"]
    half_times = [10, 60, 5*60]

    databuilder = DataBuilder(
        raw_data_folder_path="./data/cases",
        signal_features_names=signal_features_names,
        static_data_path="./data/static_data.parquet",
        static_data_names=static_features_names,
        dataset_output_folder_path="./data/datasets/signal_dataset_no_leading_time",
        sampling_time=2,
        leading_time=0,
        prediction_window_length=10*60,
        observation_window_length=5*60,
        segment_shift=30,
        half_times=half_times,
        extract_features=False,
    )

    databuilder.build()
    databuilder.build_meta()


if __name__ == "__main__":
    main()
