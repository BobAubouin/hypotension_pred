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
        dataset_output_folder_path="./data/datasets/base_dataset",
        sampling_time=2,
        leading_time=3*60,
        prediction_window_length=7*60,
        observation_window_length=5*60,
        segment_shift=30,
        half_times=half_times,
    )

    export_folder = "./data/datasets/base_dataset"
    databuilder.build()


if __name__ == "__main__":
    main()
