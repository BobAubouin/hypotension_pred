from functools import reduce
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd

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

WINDOW_SIZE_PEAK = 500  # window size for the peak detection
TRESHOLD_PEAK = 30  # threshold for the peak detection
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
NUMBER_CV_FOLD = 5  # number of cross-validation fold
RECOVERY_TIME = 10 * 60  # recovery time after the IOH (in seconds)


def detect_ioh(window: pd.Series) -> bool:
    return (window < MIN_VALUE_IOH).loc[~np.isnan(window)].all()


class DataBuilder:
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
        recovery_time: int = RECOVERY_TIME,
        max_mbp_segment: int = MAX_MBP_SEGMENT,
        min_mbp_segment: int = MIN_MBP_SEGMENT,
        treshold_peak: int = TRESHOLD_PEAK,
        max_nan_segment: float = MAX_NAN_SEGMENT,
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
        assert (
            not dataset_output_folder.exists()
        ), "Manually delete previous dataset first"
        dataset_output_folder.mkdir(parents=True)
        self.dataset_output_folder = dataset_output_folder
        # End (Generated dataset)

        # Transform data
        self.sampling_time = sampling_time
        # End (Transform data)

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
        self.n_raw_segments = 0
        self.n_selected_segments = 0
        # End (Segments parameters)

        # Features generation
        self.half_times = [half_time // sampling_time for half_time in half_times]
        # End (Features generation)

        self.window_size_peak = window_size_peak // sampling_time
        self.min_time_ioh = min_time_ioh // sampling_time
        self.recovery_time = recovery_time // sampling_time
        self.max_mbp_segment = max_mbp_segment
        self.min_mbp_segment = min_mbp_segment
        self.treshold_peak = treshold_peak
        self.max_nan_segment = max_nan_segment

    def _import_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_data = pd.read_parquet(self.raw_data_folder)
        raw_data.rename(columns=RAW_FEATURES_NAME_TO_NEW_NAME, inplace=True)
        raw_data = raw_data[self.signal_features_names + ["caseid", "Time"]]

        static_data = pd.read_parquet(self.static_data_file)[self.static_data_names]

        raw_data.sort_values(["caseid", "Time"], inplace=True)
        static_data.sort_values(["caseid"], inplace=True)
        assert len(raw_data.caseid.unique()) == len(static_data.caseid.unique())

        return raw_data, static_data

    def _preprocess_sampling(self, case_data: pd.DataFrame) -> pd.DataFrame:
        case_data.set_index("Time", inplace=True)
        case_data.index = pd.to_timedelta(case_data.index, unit="S")

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
            (case_data.mbp - rolling_mean_mbp).abs() > self.treshold_peak
        )
        case_data.sbp.mask(
            (case_data.sbp - rolling_mean_sbp).abs() > self.treshold_peak * 1.5
        )
        case_data.dbp.mask(
            (case_data.dbp - rolling_mean_dbp).abs() > self.treshold_peak
        )

        return case_data

    def _preprocess(self, case_data: pd.DataFrame) -> pd.DataFrame:
        case_data.pp_ct.fillna(0, inplace=True)

        _preprocess_functions = [self._preprocess_sampling, self._preprocess_peak]

        # NOTE: acc = accumulator
        return reduce(lambda acc, method: method(acc), _preprocess_functions, case_data)

    def _labelize(self, case_data: pd.DataFrame) -> pd.Series:
        # create the label for the case
        label_raw = (
            case_data.mbp.rolling(self.min_time_ioh, min_periods=1)
            .apply(detect_ioh)
            .fillna(0)
        )

        # Roll the window on the next self.min_time_ioh samples, see if there is a label
        label = (
            label_raw.rolling(window=self.min_time_ioh, min_periods=1)
            .max()
            .astype(bool)
            .astype(int)
        )

        return label

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
            threshold_percent = nan_ratio + (1 - nan_ratio) * MAX_NAN_SEGMENT
            threshold_n_nans = threshold_percent * self.segment_length

            if segment[signal].isna().sum() > threshold_n_nans:
                return False

        self.n_selected_segments += 1
        return True

    def _create_segment_features(
        self, segment_observation: pd.DataFrame
    ) -> pd.DataFrame:
        segment_features = pd.DataFrame()

        for half_time in self.half_times:
            for signal_name in self.signal_features_names:
                ema_name = f"{signal_name}_ema_{half_time}"
                ema = segment_observation.ewm(halflife=half_time).mean().iloc[-1]
                segment_features[ema_name] = ema

                std_name = f"{signal_name}_var_{half_time}"
                std = segment_observation.ewm(halflife=half_time).std().iloc[-1]
                segment_features[std_name] = std

        return segment_features.astype("Float32")

    def _create_segments(self, case_data: pd.DataFrame, case_id: int) -> None:
        indexes_range = range(
            0, len(case_data) - self.segment_length, self.segment_shift
        )
        segment_id = 0

        for i_time_start in indexes_range:
            self.n_raw_segments += 1

            segment = case_data.iloc[i_time_start : i_time_start + self.segment_length]

            start_time_previous_segment = max(0, i_time_start - RECOVERY_TIME)
            previous_segment = case_data.iloc[start_time_previous_segment:i_time_start]

            if not self._validate_segment(segment, previous_segment):
                continue
            segment_id += 1

            segment_observations = segment.iloc[: self.observation_window_length]
            segment_features = self._create_segment_features(segment_observations)

            segment_predictions = segment.iloc[
                (self.observation_window_length + self.leading_time)
            ]
            segment_features["label"] = (segment_predictions.label.sum() > 0).astype(int)

            segment_features["time"] = segment_predictions.index[-1]

            parquet_file = (
                self.dataset_output_folder / f"case{case_id}s{segment_id}.parquet"
            )
            try:
                segment_features.to_parquet(parquet_file, index=False)
            except:
                breakpoint()

    def build(self) -> None:
        # TODO: Build split
        # FIXME: Very slow
        raw_data, static_data = self._import_raw()

        for caseid, case_data in tqdm(raw_data.groupby("caseid")):
            case_data = self._preprocess(case_data)

            label = self._labelize(case_data)
            case_data["label"] = label

            self._create_segments(case_data, caseid)  # type: ignore

        print(
            f"All segments considered: {self.n_raw_segments}\n"
            f"Selected segments: {self.n_selected_segments}"
        )
