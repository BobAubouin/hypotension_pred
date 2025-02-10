from hp_pred.databuilder import DataBuilder


def main():
    signal_features_names = (['hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp'] +
                             #  ['mbp', 'dbp', 'sbp'] +
                             ['cycle_mean', 'cycle_systol', 'cycle_diastol'] +
                             ['cycle_std', 'cycle_pulse_pressure'] +
                             ['cycle_dPdt_max', 'cycle_dPdt_min', 'cycle_dPdt_mean', 'cycle_dPdt_std'])

    static_features_names = ["age", "bmi", "asa"]
    half_times = [60, 3*60, 10*60]

    databuilder = DataBuilder(
        raw_data_folder_path="./data/features",
        signal_features_names=signal_features_names,
        static_data_path="./data/static_data.parquet",
        static_data_names=static_features_names,
        dataset_output_folder_path="./data/datasets/30_s_waveform_dataset",
        sampling_time=30,
        mbp_column='cycle_mean',
        leading_time=2*60,
        prediction_window_length=8*60,
        observation_window_length=10*60,
        segment_shift=30,
        half_times=half_times,
        recovery_time=0,
    )

    databuilder.build()
    databuilder.build_meta()


if __name__ == "__main__":
    main()
