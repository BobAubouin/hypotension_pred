import multiprocessing as mp
import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

RAW_FEATURES_NAME_TO_NEW_NAME = {
    "Solar8000/ART_MBP": "mbp",
    "Solar8000/ART_SBP": "sbp",
    "Solar8000/ART_DBP": "dbp",
    "Solar8000/HR": "hr",
    "Solar8000/RR": "rr",
    "Solar8000/PLETH_SPO2": "spo2",
    "Solar8000/ETCO2": "etco2",
    "Orchestra/PPF20_CT": "pp_ct",
    "Primus/MAC": "mac",
}
DEVICE_NAME_TO_SAMPLING_RATE = {"mac": 7, "pp_ct": 1}
SOLAR_SAMPLING_RATE = 2

# Ratio of train samples (segments) in percent
TRAIN_RATIO = 0.7

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
MAX_NAN_SEGMENT = 0.2  # maximum acceptable value for the nan in the segment (in %)
RECOVERY_TIME = 10 * 60  # recovery time after the IOH (in seconds)
TOLERANCE_SEGMENT_SPLIT = 0.01  # tolerance for the segment split
TOLERANCE_LABEL_SPLIT = 0.005  # tolerance for the label split
N_MAX_ITER_SPLIT = 10000  # maximum number of iteration for the split


class DataBuilder:
    def _store_parameters(self):
        self.parameters_file = self.dataset_output_folder / PARAMETERS_FILENAME
        self.parameters: dict = {
            # Data description
            "signal_features_names": self.signal_features_names,
            "static_data_names": self.static_data_names,
            # Pre process parameters
            "sampling_time": self.sampling_time,
            "window_size_peak": self.window_size_peak*self.sampling_time,
            "max_mbp_segment": self.max_mbp_segment,
            "min_mbp_segment": self.min_mbp_segment,
            "threshold_peak": self.threshold_peak,
            # Segmentations parameters
            "prediction_window_length": self.prediction_window_length*self.sampling_time,
            "leading_time": self.leading_time*self.sampling_time,
            "observation_window_length": self.observation_window_length*self.sampling_time,
            "segment_shift": self.segment_shift*self.sampling_time,
            "recovery_time": self.recovery_time*self.sampling_time,
            "max_nan_segment": self.max_nan_segment,
            # Label parameters
            "min_time_ioh": self.min_time_ioh*self.sampling_time,
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
        # End (Preprocess)

        # Segments parameters
        self.leading_time = leading_time // sampling_time
        self.prediction_window_length = prediction_window_length // sampling_time
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
        self.half_times = [half_time // sampling_time for half_time in half_times]
        # End (Features generation)

        # Labelize
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
        return case_data.resample(f"{self.sampling_time}S").first()

    def _preprocess_peak(self, case_data: pd.DataFrame) -> pd.DataFrame:
        # remove too low value (before the start of the measurement)
        case_data.mbp.mask(case_data.mbp < self.min_mbp_segment, inplace=True)

        # removing the nan values at the beginning and the ending
        case_valid_mask = ~case_data.mbp.isna()
        case_data = case_data[
            (np.cumsum(case_valid_mask) > 0)
            & (np.cumsum(case_valid_mask[::-1])[::-1] > 0)
        ].copy()

        # remove peaks on the mean arterial pressure
        rolling_mean_mbp = case_data.mbp.rolling(
            window=self.window_size_peak, center=True, min_periods=10
        ).mean()
        rolling_mean_sbp = case_data.sbp.rolling(
            window=self.window_size_peak, center=True, min_periods=10
        ).mean()
        rolling_mean_dbp = case_data.dbp.rolling(
            window=self.window_size_peak, center=True, min_periods=10
        ).mean()

        # Identify peaks based on the difference from the rolling mean
        case_data.mbp.mask(
            (case_data.mbp - rolling_mean_mbp).abs() > self.threshold_peak
        )
        case_data.sbp.mask(
            (case_data.sbp - rolling_mean_sbp).abs() > self.threshold_peak * 1.5
        )
        case_data.dbp.mask(
            (case_data.dbp - rolling_mean_dbp).abs() > self.threshold_peak
        )

        return case_data

    def _preprocess(self, case_data: pd.DataFrame) -> pd.DataFrame:
        case_data.pp_ct.fillna(0, inplace=True)

        _preprocess_functions = [self._preprocess_sampling, self._preprocess_peak]

        # NOTE: acc = accumulator
        return reduce(lambda acc, method: method(acc), _preprocess_functions, case_data)

    def detect_ioh(self, window: pd.Series) -> bool:
        return (window < self.min_value_ioh).loc[~np.isnan(window)].all()

    def _labelize(self, case_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        # create the label for the case
        label_raw = (
            case_data.mbp.rolling(self.min_time_ioh, min_periods=1)
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

    def _validate_segment(
        self, segment: pd.DataFrame, previous_segment: pd.DataFrame
    ) -> bool:
        # Too low/high MBP
        mbp = segment.mbp
        if (mbp < self.min_mbp_segment).any() or (mbp > self.max_mbp_segment).any():
            return False

        # Any IOH detected in observation or leading window
        if segment.label[: (self.observation_window_length + self.leading_time)].any():
            return False

        # IOH in previous segment
        if previous_segment.label.sum() > 0:
            return False

        for signal in self.signal_features_names:
            device_rate = DEVICE_NAME_TO_SAMPLING_RATE.get(signal, SOLAR_SAMPLING_RATE)

            nan_ratio = max(0, 1 - self.sampling_time / device_rate)
            threshold_percent = nan_ratio + (1 - nan_ratio) * self.max_nan_segment
            threshold_n_nans = threshold_percent * self.observation_window_length

            if (
                segment.iloc[: self.observation_window_length][signal].isna().sum()
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
                ewm = segment_observation[signal_name].ewm(halflife=half_time)
                ema_column = signal_name + "_ema_" + str_halt_time
                std_column = signal_name + "_std_" + str_halt_time

                column_to_features[ema_column] = (ewm.mean().iloc[-1],)
                column_to_features[std_column] = (ewm.std().iloc[-1],)

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

            if not self._validate_segment(segment, previous_segment):
                continue
            segment_id += 1

            segment_observations = segment.iloc[: self.observation_window_length]
            segment_features = self._create_segment_features(segment_observations)

            segment_predictions = segment.iloc[
                (self.observation_window_length + self.leading_time):
            ]
            segment_features["label"] = (
                (segment_predictions.label.sum() > 0).astype(int),
            )

            segment_features["time"] = segment_observations.index[-1]

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

        case_df.label_id = case_df.label_id.astype(str) + "_" + case_df.caseid.astype(str)

        filename = f"case{case_id}.parquet"
        parquet_file = self.cases_folder / filename
        case_df.to_parquet(parquet_file, index=False)

    def _create_meta(self, static_data: pd.DataFrame) -> None:

        case_data = (
            pd.read_parquet(self.cases_folder)
            .groupby("caseid")
            .agg(
                segment_count=("label", "count"),
                label_count=("label", "sum"),
            )
        )

        case_ids = list(case_data.index)
        static_data = static_data[static_data.caseid.isin(case_ids)]

        train_index, cv_split_list = self._perform_split(case_data)

        case_ids_and_splits = [(case_id, "test", "test") for case_id in case_ids if case_id not in train_index]
        for i, split in enumerate(cv_split_list):
            case_ids_and_splits += [
                (case_id, "train", f"cv_{i}") for case_id in split.index
            ]

        split = pd.DataFrame.from_records(
            data=case_ids_and_splits, columns=["caseid", "split", "cv_split"]
        ).astype({"split": "category", "cv_split": "category"})
        static_data = static_data.merge(split, on="caseid")

        static_data.to_parquet(self.dataset_output_folder / "meta.parquet", index=False)

    def _perform_split(self, case_label_data: pd.DataFrame) -> tuple[list, list]:
        """Genrate a list of index for the train set. And for the cross-validation set.

        The split is done in order to have the same ratio of segments and labels in the train and test set. The CV split is also don in a balance manner. Parameters of the split are defined in the DatasetBuilder object.

        Parameters
        ----------
        case_label_data : pd.DataFrame
            Dataframe including the number of segments and labels for each case.

        Returns
        -------
        tuple[list, list]
            The first list contains the caseid of the train set. The second list contains a list of caseid for each fold of the cross-validation set.
        """
        # Train Test split
        n_iter = 0
        best_cost = np.inf
        while n_iter < self.n_max_iter_split:
            n_iter += 1

            np.random.seed(n_iter)
            split = case_label_data.index.values
            np.random.shuffle(split)

            test = case_label_data.loc[split[: int(len(split) * TRAIN_RATIO)]]
            train = case_label_data.loc[split[int(len(split) * TRAIN_RATIO):]]

            ratio_segment = (
                train["segment_count"].sum() / case_label_data["segment_count"].sum()
            )
            train_ratio_label = (
                train["label_count"].sum() / train["segment_count"].sum()
            )
            test_ratio_label = test["label_count"].sum() / test["segment_count"].sum()

            cost = (
                abs(ratio_segment - TRAIN_RATIO) / self.tolerance_segment_split
                + abs(train_ratio_label - test_ratio_label) / self.tolerance_label_split
            )

            if cost < best_cost:
                best_cost = cost
                best_iter = n_iter

            if (abs(ratio_segment - TRAIN_RATIO) < self.tolerance_segment_split) and (
                abs(train_ratio_label - test_ratio_label) < self.tolerance_label_split
            ):
                break

        np.random.seed(best_iter)
        split = case_label_data.index.values
        np.random.shuffle(split)
        train_index = split[: int(len(split) * TRAIN_RATIO)]

        train = case_label_data.loc[case_label_data.index.isin(train_index)]
        test = case_label_data.loc[~case_label_data.index.isin(train_index)]

        print(
            f"Train : {train['segment_count'].sum() / case_label_data['segment_count'].sum()*100:.2f} % of segments, {train['label_count'].sum() / train['segment_count'].sum()*100:.2f} % of labels"
        )
        print(
            f"Test : {test['segment_count'].sum() / case_label_data['segment_count'].sum()*100:.2f} % of segments, {test['label_count'].sum() / test['segment_count'].sum()*100:.2f} % of labels"
        )

        # Cross-validation split

        ratio_split = 1 / self.number_cv_splits
        ratio_label = (
            train["label_count"].sum() / train["segment_count"].sum()
        )
        nb_iter = 0
        best_cost = np.inf
        best_split = None
        while True:
            nb_iter += 1
            if nb_iter > self.n_max_iter_split:
                break
            np.random.seed(nb_iter)
            split = train.index.values
            np.random.shuffle(split)

            index_list = np.array_split(split, self.number_cv_splits)

            split_list = [train.loc[index] for index in index_list]

            ratio_segment_list = [
                split["segment_count"].sum() / train["segment_count"].sum()
                for split in split_list
            ]
            ratio_label_list = [
                split["label_count"].sum() / split["segment_count"].sum()
                for split in split_list
            ]

            cost = sum(
                [abs(ratio - ratio) / self.tolerance_segment_split for ratio in ratio_segment_list]
            ) + sum([abs(ratio - ratio_label) / self.tolerance_label_split for ratio in ratio_label_list])

            if cost < best_cost:
                best_cost = cost
                best_split = nb_iter

            if (
                max([abs(ratio - ratio) / self.tolerance_segment_split for ratio in ratio_segment_list])
                < self.tolerance_segment_split
            ) and (
                max([abs(ratio - ratio_label) / self.tolerance_label_split for ratio in ratio_label_list])
                < self.tolerance_label_split
            ):
                break

        np.random.seed(best_split)
        split = train.index.values
        np.random.shuffle(split)
        index_list = []
        for i in range(self.number_cv_splits):
            index_list.append(
                split[
                    int(len(split) * i * ratio_split): int(
                        len(split) * (i + 1) * ratio_split
                    )
                ]
            )

        cv_split_list = [train.loc[index] for index in index_list]

        # print the result of the split
        print(f"Cross-validation split : {self.number_cv_splits} splits")
        for i, split in enumerate(cv_split_list):
            print(
                f"split {i} : {split['segment_count'].sum():,d} segments, "
                f"{split['label_count'].sum():,d} labels, "
                f"{split['label_count'].sum() / split['segment_count'].sum():.2%} ratio label"
            )

        return train_index.tolist(), cv_split_list

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
        with mp.Pool() as pool:
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
