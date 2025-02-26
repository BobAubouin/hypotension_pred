import json
import multiprocessing as mp
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from sklearn.linear_model import LinearRegression

from hp_pred.split import (
    compute_ratio_label,
    compute_ratio_segment,
    create_balanced_split,
    create_cv_balanced_split,
)

RAW_FEATURES_NAME_TO_NEW_NAME = {
    "Solar8000/ART_MBP": "mbp",
    "Solar8000/ART_SBP": "sbp",
    "Solar8000/ART_DBP": "dbp",
    "Solar8000/HR": "hr",
    "Solar8000/RR_CO2": "rr",
    "Solar8000/PLETH_SPO2": "spo2",
    "Solar8000/ETCO2": "etco2",
    "Solar8000/BT": "body_temp",
    "Orchestra/PPF20_CT": "pp_ct",
    "Orchestra/RFTN20_CT": "rf_ct",
    "Orchestra/VASO_RATE": "vaso_rate",
    "Orchestra/PHEN_RATE": "phen_rate",
    "Orchestra/NEPI_RATE": "nepi_rate",
    "Orchestra/EPI_RATE": "epi_rate",
    "Orchestra/DOPA_RATE": "dopa_rate",
    "Orchestra/DOBU_RATE": "dobu_rate",
    "Orchestra/DTZ_RATE": "dtz_rate",
    "Orchestra/NTG_RATE": "ntg_rate",
    "Orchestra/NPS_RATE": "nps_rate",
    "Primus/MAC": "mac",
}

INTERVENTION_DRUGS = [
    "pp_ct",
    "rf_ct",
    "vaso_rate",
    "phen_rate",
    "nepi_rate",
    "epi_rate",
    "dopa_rate",
    "dobu_rate",
    "dtz_rate",
    "ntg_rate",
    "nps_rate",
    "mac",
]

VASSOPRESSOR_DRUGS = [
    "vaso_rate",
    "phen_rate",
    "nepi_rate",
    "epi_rate",
    "dopa_rate",
    "dobu_rate",
    "nps_rate",
]


DEVICE_NAME_TO_SAMPLING_RATE = {"mac": 7, "pp_ct": 1}
SOLAR_SAMPLING_RATE = 2

CASE_SUBFOLDER_NAME = "cases"
PARAMETERS_FILENAME = "DatasetBuilder_parameters.json"

# Defaults parameters for DatasetBuilder
WINDOW_SIZE_PEAK = 500  # window size for the peak detection
THRESHOLD_PEAK = 30  # threshold for the peak detection
MIN_TIME_IOH = 60  # minimum time for the IOH to be considered as IOH (in seconds)
MIN_VALUE_IOH = (
    65  # minimum value for the mean arterial pressure to be considered as IOH (in mmHg)
)
MIN_MBP_SEGMENT = (
    40  # minimum acceptable value for the mean arterial pressure (in mmHg)
)
MAX_MBP_SEGMENT = (
    150  # maximum acceptable value for the mean arterial pressure (in mmHg)
)
MAX_NAN_SEGMENT = 0.50  # maximum acceptable value for the nan in the segment (in %)
RECOVERY_TIME = 10 * 60  # recovery time after the IOH (in seconds)
TOLERANCE_SEGMENT_SPLIT = 0.01  # tolerance for the segment split
TOLERANCE_LABEL_SPLIT = 0.005  # tolerance for the label split
N_MAX_ITER_SPLIT = 500_000  # maximum number of iteration for the split
SMOOTH_PERIOD = 40  # period for the rolling mean in seconds


class DataBuilderReg:
    def _store_parameters(self):
        self.parameters_file = self.dataset_output_folder / PARAMETERS_FILENAME
        self.parameters: dict = {
            # Data description
            "signal_features_names": self.signal_features_names,
            "static_data_names": self.static_data_names,
            # Pre process parameters
            "sampling_time": self.sampling_time,
            "window_size_peak": self.window_size_peak * self.sampling_time,
            "max_mbp_segment": self.max_mbp_segment,
            "min_mbp_segment": self.min_mbp_segment,
            "threshold_peak": self.threshold_peak,
            # Segmentations parameters
            "prediction_window_length": self.prediction_window_length
            * self.sampling_time,
            "leading_time": self.leading_time * self.sampling_time,
            "observation_window_length": self.observation_window_length
            * self.sampling_time,
            "segment_shift": self.segment_shift * self.sampling_time,
            "recovery_time": self.recovery_time * self.sampling_time,
            "max_nan_segment": self.max_nan_segment,
            # Label parameters
            "mbp_column": self.mbp_column,
            "min_time_ioh": self.min_time_ioh * self.sampling_time,
            "min_value_ioh": self.min_value_ioh,
            # Features parameters
            "half_times": [half_time * self.sampling_time for half_time in self.half_times],
            # Split parameters
            "tolerance_segment_split": self.tolerance_segment_split,
            "tolerance_label_split": self.tolerance_label_split,
            "n_max_iter_split": self.n_max_iter_split,
            "number_cv_splits": self.number_cv_splits,
            # folder information
            "raw_data_folder_path": str(self.raw_data_folder),
            "static_data_path": str(self.static_data_file),
            "dataset_output_folder_path": str(self.dataset_output_folder),
            "extract_features": self.extract_features,
            "smooth_period": self.smooth_period * self.sampling_time,
        }

    def __init__(
        self,
        raw_data_folder_path: str,
        signal_features_names: list[str],
        static_data_path: str,
        static_data_names: list[str],
        dataset_output_folder_path: str,
        sampling_time: int,
        leading_time: int,
        prediction_window_length: int,
        observation_window_length: int,
        segment_shift: int,
        half_times: list[int],
        window_size_peak: int = WINDOW_SIZE_PEAK,
        mbp_column: str = "mbp",
        min_time_ioh: int = MIN_TIME_IOH,
        min_value_ioh: float = MIN_VALUE_IOH,
        recovery_time: int = RECOVERY_TIME,
        max_mbp_segment: int = MAX_MBP_SEGMENT,
        min_mbp_segment: int = MIN_MBP_SEGMENT,
        threshold_peak: int = THRESHOLD_PEAK,
        max_nan_segment: float = MAX_NAN_SEGMENT,
        tolerance_segment_split: float = TOLERANCE_SEGMENT_SPLIT,
        tolerance_label_split: float = TOLERANCE_LABEL_SPLIT,
        n_max_iter_split: int = N_MAX_ITER_SPLIT,
        number_cv_splits: int = 3,
        extract_features: bool = True,
        smooth_period: int = SMOOTH_PERIOD,
    ) -> None:
        # Raw data
        raw_data_folder = Path(raw_data_folder_path)
        assert raw_data_folder.exists()
        assert any(file.suffix == ".parquet" for file in raw_data_folder.iterdir())
        self.raw_data_folder = raw_data_folder
        self.signal_features_names = signal_features_names

        static_data_file = Path(static_data_path)
        assert static_data_file.exists()
        self.static_data_file = static_data_file
        self.static_data_names = static_data_names + ["caseid"]
        # End (Raw data)

        # Generated dataset
        dataset_output_folder = Path(dataset_output_folder_path)
        self.dataset_output_folder = dataset_output_folder
        self.cases_folder = self.dataset_output_folder / CASE_SUBFOLDER_NAME
        # End (Generated dataset)

        # Preprocess
        self.sampling_time = sampling_time
        self.window_size_peak = window_size_peak // sampling_time
        self.max_mbp_segment = max_mbp_segment
        self.min_mbp_segment = min_mbp_segment
        self.threshold_peak = threshold_peak
        self.smooth_period = smooth_period // sampling_time
        # End (Preprocess)

        # Segments parameters
        self.leading_time = leading_time // sampling_time - 1
        self.prediction_window_length = prediction_window_length // sampling_time + 1
        self.observation_window_length = observation_window_length // sampling_time
        self.segment_shift = segment_shift // sampling_time
        self.segment_length = (
            self.observation_window_length
            + self.leading_time
            + self.prediction_window_length
        )
        self.recovery_time = recovery_time // sampling_time
        self.max_nan_segment = max_nan_segment
        # End (Segments parameters)

        # Features generation
        self.extract_features = extract_features
        self.half_times = [half_time // sampling_time for half_time in half_times]
        # End (Features generation)

        # Labelize
        self.mbp_column = mbp_column
        self.min_time_ioh = min_time_ioh // sampling_time
        self.min_value_ioh = min_value_ioh
        # End (Labelize)

        # Split
        self.tolerance_segment_split = tolerance_segment_split
        self.tolerance_label_split = tolerance_label_split
        self.n_max_iter_split = n_max_iter_split
        self.number_cv_splits = number_cv_splits
        # End (Split)

        self._store_parameters()

    def _import_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_data = pd.read_parquet(self.raw_data_folder)
        raw_data.rename(columns=RAW_FEATURES_NAME_TO_NEW_NAME, inplace=True)
        raw_data = raw_data[self.signal_features_names + ["caseid", "Time"]]

        static_data = pd.read_parquet(self.static_data_file)[self.static_data_names]
        static_data = static_data.loc[:, ~static_data.columns.duplicated()].copy()
        static_data = static_data[static_data.caseid.isin(raw_data.caseid.unique())]
        assert len(raw_data.caseid.unique()) == len(static_data.caseid.unique())

        raw_data.Time = pd.to_timedelta(raw_data.Time, unit="s")
        raw_data.set_index(["caseid", "Time"], inplace=True)
        static_data.set_index(["caseid"])

        raw_data.sort_index(inplace=True)
        static_data.sort_index(inplace=True)

        return raw_data, static_data

    def _preprocess_sampling(self, case_data: pd.DataFrame) -> pd.DataFrame:
        if self.sampling_time == 30:
            # to perform as in Grenoble dataset
            return case_data.resample(f"{self.sampling_time}S", closed='right', label='right').median()
        return case_data.resample(f"{self.sampling_time}S").last()

    def _preprocess_peak(self, case_data: pd.DataFrame) -> pd.DataFrame:
        # remove too low value (before the start of the measurement)
        if 'mbp' not in case_data.columns:
            return case_data
        case_data.mbp.mask(case_data.mbp < self.min_mbp_segment, inplace=True)
        case_data.mbp.mask(case_data.mbp > self.max_mbp_segment, inplace=True)

        # removing the nan values at the beginning and the ending
        case_valid_mask = ~case_data.mbp.isna()
        case_data = case_data[
            (np.cumsum(case_valid_mask) > 0)
            & (np.cumsum(case_valid_mask[::-1])[::-1] > 0)
        ].copy()

        # remove peaks on the mean arterial pressure
        # rolling_mean_mbp = case_data.mbp.rolling(
        #     window=self.window_size_peak, center=True, min_periods=10
        # ).mean()
        # rolling_mean_sbp = case_data.sbp.rolling(
        #     window=self.window_size_peak, center=True, min_periods=10
        # ).mean()
        # rolling_mean_dbp = case_data.dbp.rolling(
        #     window=self.window_size_peak, center=True, min_periods=10
        # ).mean()

        # # Identify peaks based on the difference from the rolling mean
        # case_data.mbp.mask(
        #     (case_data.mbp - rolling_mean_mbp).abs() > self.threshold_peak,
        #     inplace=True,
        # )
        # case_data.sbp.mask(
        #     (case_data.sbp - rolling_mean_sbp).abs() > self.threshold_peak * 1.5,
        #     inplace=True,
        # )
        # case_data.dbp.mask(
        #     (case_data.dbp - rolling_mean_dbp).abs() > self.threshold_peak,
        #     inplace=True,
        # )

        return case_data

    def _preprocess_smooth(self, case_data: pd.DataFrame) -> pd.DataFrame:
        signals_to_smooth = [sign for sign in self.signal_features_names if sign != 'pp_ct']
        case_data[signals_to_smooth] = case_data[signals_to_smooth].rolling(
            window=self.smooth_period, min_periods=1).mean()
        return case_data

    def _preprocess_fillna(self, case_data: pd.DataFrame) -> pd.DataFrame:
        for drug in INTERVENTION_DRUGS:
            if drug in case_data.columns:
                if drug != 'mac':
                    case_data[drug].ffill(inplace=True)
                case_data[drug].fillna(0, inplace=True)
        if 'mbp' not in case_data.columns:
            return case_data
        case_data.mbp = case_data.mbp.interpolate()
        case_data.sbp = case_data.sbp.interpolate()
        case_data.dbp = case_data.dbp.interpolate()

        return case_data

    def _preprocess(self, case_data: pd.DataFrame) -> pd.DataFrame:

        _preprocess_functions = [self._preprocess_sampling, self._preprocess_peak,
                                 self._preprocess_smooth, self._preprocess_fillna]

        # NOTE: acc = accumulator
        return reduce(lambda acc, method: method(acc), _preprocess_functions, case_data)

    def detect_ioh(self, window: pd.Series) -> bool:
        return (window < self.min_value_ioh).loc[~np.isnan(window)].all()

    def _labelize(self, case_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        # create the label for the case
        label_raw = (
            case_data[self.mbp_column].rolling(self.min_time_ioh, min_periods=1)
            .apply(self.detect_ioh)
            .fillna(0)
        )

        # Roll the window on the next self.min_time_ioh samples, see if there is a label
        label = (
            label_raw.rolling(window=self.min_time_ioh, min_periods=1)
            .max()
            .shift(-self.min_time_ioh + 1, fill_value=0)
        )

        label_id = label.diff().clip(lower=0).cumsum().fillna(0)
        label_id = label_id.astype(int)
        label_id[label == 0] = np.nan

        return label, label_id

    def detect_intervention_hypo(self, segment: pd.DataFrame) -> bool:
        # Remove part of the segment after label==1 to only consider intervention before possible IOH
        if segment.label.sum() > 0:
            time_label = segment.label.idxmax()
            if time_label > segment.index[0]:
                time_label -= pd.Timedelta(seconds=self.sampling_time)
            segment = segment.loc[: time_label]
            hypo = True
        else:
            hypo = False
        # Test if there is an intervention
        for drug in INTERVENTION_DRUGS:
            if drug not in segment.columns:
                continue
            if hypo and drug in VASSOPRESSOR_DRUGS:
                if segment[drug].iloc[0] - segment[drug].min() > 0:
                    return True
            elif hypo and drug not in VASSOPRESSOR_DRUGS:
                if segment[drug].iloc[0] - segment[drug].max() < 0:
                    return True
            elif not hypo and drug in VASSOPRESSOR_DRUGS:
                if segment[drug].iloc[0] - segment[drug].max() < 0:
                    return True
            elif not hypo and drug not in VASSOPRESSOR_DRUGS:
                if segment[drug].iloc[0] - segment[drug].min() > 0:
                    return True
        return False

    def detect_intervention(self, segment: pd.DataFrame) -> bool:
        # Test if there is an intervention
        for drug in INTERVENTION_DRUGS:
            if drug not in segment.columns:
                continue
            if segment[drug].max() == 0:
                continue
            if (segment[drug].max() - segment[drug].min())/segment[drug].max() > 0.05:
                return True
        return False

    def _validate_segment(
        self, segment: pd.DataFrame, previous_segment: pd.DataFrame
    ) -> bool:
        # Too low/high MBP
        mbp = segment[self.mbp_column]
        if (mbp < self.min_mbp_segment).any() or (mbp > self.max_mbp_segment).any():
            return False

        # Any IOH detected in observation or leading window
        # if segment.label[: (self.observation_window_length + self.leading_time)].any():
        #     return False

        # IOH in previous segment
        if previous_segment.label.sum() > 0:
            return False

        for signal in self.signal_features_names:
            if signal in ["mac", "pp_ct"]:
                continue

            threshold_percent = self.max_nan_segment
            threshold_n_nans = threshold_percent * self.segment_length

            if (
                segment[signal].isnull().sum()
                > threshold_n_nans
            ):
                return False

        return True

    def _create_segment_features(
        self, segment_observation: pd.DataFrame
    ) -> pd.DataFrame:
        column_to_features: dict[str, tuple[float]] = {}

        for half_time in self.half_times:
            str_halt_time = str(half_time * self.sampling_time)
            for signal_name in self.signal_features_names:
                constant_features = signal_name + "_constant_" + str_halt_time
                slope_features = signal_name + "_slope_" + str_halt_time
                std_features = signal_name + "_std_" + str_halt_time

                if half_time < 2:
                    half_time = 2
                else:
                    model = LinearRegression()
                    X = np.arange(-half_time, 0).reshape(-1, 1)
                    y = segment_observation[signal_name].iloc[-half_time:]
                    # remove nan from y
                    X = X[~y.isna()]
                    y = y[~y.isna()]
                    if len(y) == 0:
                        column_to_features[constant_features] = (np.nan,)
                        column_to_features[slope_features] = (np.nan,)
                        column_to_features[std_features] = (np.nan,)
                        continue
                    if len(y) == 1:
                        column_to_features[constant_features] = (y.iloc[0],)
                        column_to_features[slope_features] = (0,)
                        column_to_features[std_features] = (0,)
                        continue
                    model.fit(X, y)
                    column_to_features[constant_features] = (model.intercept_,)
                    column_to_features[slope_features] = (model.coef_[0],)
                    y_pred = model.predict(np.arange(-half_time, 0).reshape(-1, 1))
                    error = segment_observation[signal_name].iloc[-half_time:] - y_pred
                    column_to_features[std_features] = (
                        error.std()
                    )

                # ema_column = signal_name + "_ema_" + str_halt_time
                # std_column = signal_name + "_std_" + str_halt_time

                # if half_time == 0:
                #     column_to_features[ema_column] = (
                #         segment_observation[signal_name].iloc[-1],
                #     )
                #     column_to_features[std_column] = (
                #         segment_observation[signal_name].diff().iloc[-1],
                #     )
                # else:
                #     ewm = segment_observation[signal_name].ewm(halflife=half_time)

                #     column_to_features[ema_column] = (ewm.mean().iloc[-1],)
                #     column_to_features[std_column] = (ewm.std().iloc[-1],)
        # add last value as baseline feature
        column_to_features["last_map_value"] = (segment_observation[self.mbp_column].iloc[-1],)
        return pd.DataFrame(column_to_features, dtype="Float32")

    def _create_segments(self, case_data: pd.DataFrame, case_id: int) -> None:
        indexes_range = range(
            0, len(case_data) - self.segment_length, self.segment_shift
        )
        segment_id = 0
        list_of_segments = []
        for i_time_start in indexes_range:
            segment = case_data.iloc[i_time_start: i_time_start + self.segment_length]

            start_time_previous_segment = max(0, i_time_start - self.recovery_time)
            previous_segment = case_data.iloc[start_time_previous_segment:i_time_start]
            segment_observations = segment.iloc[: self.observation_window_length]

            if not self._validate_segment(segment_observations, previous_segment):
                continue
            segment_id += 1

            if self.extract_features:
                segment_features = self._create_segment_features(segment_observations)
            else:
                segment_features = segment_observations[self.signal_features_names].copy()
                segment_features.index = range(len(segment_features))
                stacked_df = segment_features.stack().reset_index()
                stacked_df.columns = ['timestamp', 'Signal', 'Value']
                segment_features = stacked_df.set_index(['Signal', 'timestamp']).sort_index().transpose()

            segment_predictions = segment.iloc[
                (self.observation_window_length + self.leading_time):
            ]
            segment_features["label"] = (
                (segment_predictions.label.sum() > 0).astype(int),
            )

            # add future map values
            for i in range(0, self.prediction_window_length):
                segment_features[f"future_map_value_{i}"] = (
                    segment_predictions[self.mbp_column].iloc[i],
                )

            segment_features["time"] = segment_observations.index[-1]
            segment_features["intervention"] = self.detect_intervention(
                segment.iloc[self.observation_window_length:]
            )

            segment_features["intervention_hypo"] = self.detect_intervention_hypo(
                segment.iloc[self.observation_window_length:]
            )

            segment_features["ioh_at_time_t"] = segment_observations.label.iloc[-1]
            segment_features["ioh_in_leading_time"] = (
                segment[self.observation_window_length:self.observation_window_length+self.leading_time].label.sum() > 0
            )
            #     return False

            if segment_features.label.iloc[0] == 1:
                segment_features["time_before_IOH"] = (
                    segment_predictions.label.idxmax() - segment_observations.index[-1]
                ).seconds
                segment_features["label_id"] = segment_predictions.loc[
                    segment_predictions.label.idxmax()
                ].label_id
            else:
                segment_features["time_before_IOH"] = np.nan
                segment_features["label_id"] = np.nan

            segment_features["caseid"] = case_id

            list_of_segments.append(segment_features)

        if len(list_of_segments) == 0:
            return
        case_df = pd.concat(list_of_segments, axis=0, ignore_index=True)

        case_df.label_id = (
            case_df.label_id.astype(str) + "_" + case_df.caseid.astype(str)
        )

        filename = f"case{int(case_id):04d}.parquet"
        parquet_file = self.cases_folder / filename
        case_df.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in case_df.columns]
        for col in case_df.columns:
            if col.endswith('_'):
                # remove the '_' at the end of the column name
                case_df.rename(columns={col: col[:-1]}, inplace=True)
        case_df.to_parquet(parquet_file, index=False)

    def _create_meta(self, static_data: pd.DataFrame) -> None:

        # Index: case ID
        # Value:
        #  - segment_count: number of segment of the case IDs
        #  - label_count: number of positive of the case IDs
        label_stats = (
            pd.read_parquet(self.cases_folder, columns=["label", "caseid"])
            .groupby("caseid")
            .agg(
                # case_id=("caseid", "first"),
                segment_count=("label", "count"),
                label_count=("label", "sum"),
            )
        )

        case_ids = list(label_stats.index)
        static_data = static_data[static_data.caseid.isin(case_ids)]

        train_index, cv_split_list = self._perform_split(label_stats)

        case_ids_and_splits = [
            (case_id, "test", "test")
            for case_id in case_ids
            if case_id not in train_index
        ]
        for i, split in enumerate(cv_split_list):
            case_ids_and_splits += [
                (case_id, "train", f"cv_{i}") for case_id in split.index
            ]

        split = pd.DataFrame.from_records(
            data=case_ids_and_splits, columns=["caseid", "split", "cv_split"]
        ).astype({"split": "category", "cv_split": "category"})
        static_data = static_data.merge(split, on="caseid")

        static_data.to_parquet(self.dataset_output_folder / "meta.parquet", index=False)

    def _perform_split(self, label_stats: pd.DataFrame) -> tuple[list[int], list[pd.DataFrame]]:
        """Genrate a list of index for the train set. And for the cross-validation set.

        The split is done in order to have the same ratio of segments and labels in the train and test set. The CV split is also don in a balance manner. Parameters of the split are defined in the DatasetBuilder object.

        Parameters
        ----------
        label_stats : pd.DataFrame
            Index by case IDs, it has "segment_count" and "label_count" columns:
              - segment_count: number of segment of the case IDs
              - label_count: number of positive of the case IDs

        Returns
        -------
        tuple[list, list]
            The first list contains the caseid of the train set. The second list contains a list of caseid for each fold of the cross-validation set.
        """
        train_label_stats, test_label_stats = create_balanced_split(
            label_stats,
            self.tolerance_segment_split,
            self.tolerance_label_split,
            self.n_max_iter_split,
        )

        train_ratio_segment = compute_ratio_segment(train_label_stats, label_stats)
        train_ratio_label = compute_ratio_label(train_label_stats)
        print(
            f"Train : {train_label_stats.segment_count.sum():,d} segments "
            f"({train_ratio_segment:.2%}), "
            f" {train_ratio_label:.2%} of labels"
        )

        test_ratio_segment = compute_ratio_segment(test_label_stats, label_stats)
        test_ratio_label = compute_ratio_label(test_label_stats)
        print(
            f"Test : {test_label_stats.segment_count.sum():,d} segments "
            f"({test_ratio_segment:.2%}), "
            f"{test_ratio_label:.2%} of labels"
        )

        train_cv_label_stats_splits = create_cv_balanced_split(
            train_label_stats,
            train_ratio_segment,
            self.number_cv_splits,
            self.tolerance_segment_split,
            self.tolerance_label_split,
            self.n_max_iter_split,
        )

        print(f"Cross-validation split : {self.number_cv_splits} splits")
        for i, split_label_stats in enumerate(train_cv_label_stats_splits):
            split_label_ratio = compute_ratio_label(split_label_stats)
            print(
                f"split {i} : {split_label_stats.segment_count.sum():,d} segments, "
                f"{split_label_stats.label_count.sum():,d} labels, "
                f"{split_label_ratio:.2%} ratio label"
            )

        return train_label_stats.index.to_list(), train_cv_label_stats_splits

    def _process_case(self, param) -> None:
        caseid, case_data = param
        case_data = case_data.reset_index("caseid", drop=True)
        case_data = self._preprocess(case_data)

        label, label_id = self._labelize(case_data)
        case_data["label"] = label
        case_data["label_id"] = label_id

        self._create_segments(case_data, caseid)

    def _dump_dataset_parameter(self) -> None:
        with open(self.parameters_file, mode="w", encoding="utf-8") as file:
            json.dump(self.parameters, file, indent=2)

    def _must_be_built(self) -> bool:
        # build the dataset if the dataset does not exist already.
        if not self.dataset_output_folder.exists():
            return True

        with open(self.parameters_file, mode="r", encoding="utf-8") as file:
            parameters = json.load(file)

        # build the dataset if the existing dataset has been build with different
        # parameters.
        return self.parameters != parameters

    def build(self) -> None:
        if not self._must_be_built():
            print(
                f"The same dataset is already built with the same parameters in folder "
                f"{self.dataset_output_folder}."
            )
            print("Dataset build aborted")
            return

        self.cases_folder.mkdir(parents=True, exist_ok=True)

        print("Loading raw data...")
        raw_data, static_data = self._import_raw()

        print("Segmentation...")
        with mp.Pool():
            process_map(
                self._process_case,
                raw_data.groupby("caseid", as_index=False),
                total=len(static_data),
                chunksize=1,
            )

        self._dump_dataset_parameter()

    def build_meta(self) -> None:

        static_data = pd.read_parquet(self.static_data_file)
        self._create_meta(static_data)
        self._dump_dataset_parameter()

    @classmethod
    def from_json(cls, dataset_folder: Path):
        filename = dataset_folder / PARAMETERS_FILENAME
        with open(filename, mode="r", encoding="utf-8") as file:
            parameters = json.load(file)

        return cls(**parameters)
