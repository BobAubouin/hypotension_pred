import argparse
import asyncio
import datetime
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from hp_pred.constants import VITAL_API_BASE_URL
from hp_pred.data_retrieve_async import retrieve_tracks_raw_data_async
from hp_pred.tracks_config import (
    DEVICE_NAME_TO_SAMPLING_RATE,
    STATIC_DATA_NAMES,
    TRACKS_CONFIG,
    TrackConfig,
    SAMPLING_TIME
)

TRACKS_META_URL = f"{VITAL_API_BASE_URL}/trks"
CASE_INFO_URL = f"{VITAL_API_BASE_URL}/cases"

# Filter constants
TRACK_NAME_MBP = "Solar8000/ART_MBP"
# Duration in seconds
CASEEND_CASE_THRESHOLD = 3600
FORBIDDEN_OPNAME_CASE = "transplant"
PERCENT_MISSING_DATA_THRESHOLD = 0.5
AGE_CASE_THRESHOLD = 18
BLOOD_LOSS_THRESHOLD = 200  # ml


def parse() -> tuple[str, Path]:
    parser = argparse.ArgumentParser(
        description="Download the VitalDB data for hypertension prediction."
    )

    log_level_names = list(logging.getLevelNamesMapping().keys())
    parser.add_argument(
        "-l",
        "--log_level_name",
        type=str,
        default="INFO",
        choices=log_level_names,
        help="The logger level name to generate logs. (default: %(default)s)",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="data",
        help="The folder to store the data and logs. (default: %(default)s)",
    )

    args = parser.parse_args()

    log_level_name = args.log_level_name
    output_folder = Path(args.output_folder)

    return log_level_name, output_folder


def setup_logger(output_folder: Path, log_level_name: str):
    global logger
    logger = logging.getLogger("log")

    log_level = logging.getLevelNamesMapping()[log_level_name]
    logger.setLevel(log_level)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_formatter = logging.Formatter(log_format)

    # Console handler, log everything.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)

    # File handler
    log_file_name = output_folder / f"run-{datetime.datetime.now()}.log"
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_formatter)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def get_track_names(tracks: list[TrackConfig] = TRACKS_CONFIG) -> list[str]:
    """
    Get a list of track names from a list of dictionnaries (TrackConfig)

    Args:
        tracks (list[TrackConfig], optional): List of config, 1 config for each device.
            Defaults to TRACKS_CONFIG.

    Returns:
        list[str]: List of the track names.
    """
    track_names = [
        f"{track['name']}/{track_name}"
        for track in tracks
        for track_name in track["tracks"]
    ]

    info_track_names = ", ".join(track_name for track_name in track_names)
    logger.info(f"{info_track_names} track names will be added to the dataset\n")

    return track_names


def filter_case_ids(cases: pd.DataFrame, tracks_meta: pd.DataFrame) -> list[int]:
    """
    Filter the cases to download based on some criteria:
        - The case should have the MBP track
        - The patient should be at least 18 years old
        - No EMOP
        - The number of seconds should be more than a threshold
        - One operation is forbidden
        - The blood loss should be less than a threshold

    Note: This filter is not configurable on purpose, it is meant to be static.

    Args:
        cases (pd.DataFrame): Dataframe of the VitalDB cases
        tracks_meta (pd.DataFrame): The meta-data of the cases.

    Returns:
        list[int]: List of the valid case IDs.
    """
    cases_with_mbp = pd.merge(
        tracks_meta.query(f"tname == '{TRACK_NAME_MBP}'"),
        cases,
        on="caseid",
    )

    filtered_unique_case_ids = (
        cases_with_mbp[
            (cases_with_mbp.age > AGE_CASE_THRESHOLD)
            & (cases_with_mbp.caseend > CASEEND_CASE_THRESHOLD)
            & (~cases_with_mbp.opname.str.contains(FORBIDDEN_OPNAME_CASE, case=False))
            & (cases_with_mbp.emop == 0)
            & ((cases_with_mbp.intraop_ebl < BLOOD_LOSS_THRESHOLD) | (cases_with_mbp.intraop_ebl.isna()))
        ]
        .caseid.unique()
        .tolist()
    )

    return filtered_unique_case_ids


def retrieve_tracks_raw_data(tracks_meta: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Use the `hp_pred.data_retrieve_async` module to get new data.

    Args:
        tracks_meta (pd.DataFrame): The tracks' meta-data (track URL and case ids) to
        retrieve.

    Returns:
        list[pd.DataFrame]: The tracks data for each case ID.
    """
    tracks_url_and_case_id = [
        (f"/{track.tid}", int(track.caseid))  # type: ignore
        for track in tracks_meta.itertuples(index=False)
    ]

    logger.debug("Start retrieving data from VitalDB API")
    tracks_raw_data = asyncio.run(
        retrieve_tracks_raw_data_async(tracks_url_and_case_id)
    )
    logger.debug("Done retrieving data from VitalDB API")

    logger.debug("Start gathering raw track data by case ID.")
    case_id_to_track_raw_data_list: dict[int, list[pd.DataFrame]] = defaultdict(list)
    for track_raw_data in tracks_raw_data:
        # Retrieve the case_id and check that everyrow has the same in a single df.
        case_id = track_raw_data.caseid.iloc[0]
        assert (track_raw_data.caseid == case_id).all()

        case_id_to_track_raw_data_list[case_id].append(track_raw_data)
    logger.debug("Case ID coherence done.")

    tracks_raw_data_gathered = [
        pd.concat(track_raw_data_list)
        for track_raw_data_list in case_id_to_track_raw_data_list.values()
    ]
    logger.info("Done gathering raw track data by case ID.")
    return tracks_raw_data_gathered


def format_track_raw_data_wav(track_raw_data: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def format_track_raw_data_num(track_raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the track's raw data according to the Time column. The Time column is rounded
    and we group the different values with the same rounded Time value.

    Args:
        track_raw_data (pd.DataFrame): Raw data retrieved from the VitalDB API.

    Returns:
        pd.DataFrame: Track data with integer Time and fewer NaN.
    """
    track_raw_data.Time = (track_raw_data.Time / SAMPLING_TIME).astype(int)

    return track_raw_data.groupby("Time", as_index=False).first()


def format_time_track_raw_data(track_raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the track's raw data. It chooses between the numeric and the wave formats.

    Args:
        track_raw_data (pd.DataFrame): Raw data retrieved from the VitalDB API.

    Returns:
        pd.DataFrame: Track data with integer Time and fewer NaN.
    """
    track_raw_data = track_raw_data.astype(
        {
            column_name: "float32"
            for column_name in track_raw_data.columns
            if column_name != "Time"
        }
    )

    return (
        format_track_raw_data_wav(track_raw_data)
        if track_raw_data.Time.hasnans
        else format_track_raw_data_num(track_raw_data)
    )


def _has_enough_data(
    track: pd.Series,
    sampling_rate: int,
    percent_missing_data_threshold: float = PERCENT_MISSING_DATA_THRESHOLD,
) -> bool:
    """
    Check if there is enough data in the track data.
    The tracks have different sampling time (lower of the second here)
    For example, if track has 200 rows with sampling time of .5, at best there would be
    100 rows with value (not NaN).
    An actual track might have different amount of values, lesser than 100.
    We want to make sure those tracks have enough data. So we fix a percentage over this
    max number of values (100).
    Let's say we are okay to have 60% of the max number of values (60).
    Then we make sure that we have 60 rows with values.
    Args:
        track (pd.Series): The data values.
        sampling_rate (int): The sampling rate of the track.
        percent_missing_data_threshold (float, optional): How much missng data we allow.
            Defaults to PERCENT_MISSING_DATA_THRESHOLD.
    Returns:
        bool: There is enough data (True), are too much NaN values (False).
    """
    max_number_values = len(track) / sampling_rate
    acceptable_number_values = (1 - percent_missing_data_threshold) * max_number_values
    number_values = track.notna().sum()

    has_not_enough_data = number_values < acceptable_number_values

    if has_not_enough_data:
        return False

    return True


def post_process_track(
    track: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Check if the tracks has enough data and add some static features.

    Args:
        track (pd.DataFrame): Tracks corresponding to one case ID.
        cases (pd.DataFrame): All cases information.

    Returns:
        pd.DataFrame | None: If the track data are not suitable, else return the track
            data with its static data from cases.
    """

    track_names = [
        column for column in track.columns if column not in ["Time", "caseid"]
    ]

    # Ensure data the patient has enough data.
    for track_name in track_names:
        device_name = track_name.split("/")[0]
        if not device_name in DEVICE_NAME_TO_SAMPLING_RATE:
            continue

        sampling_rate = DEVICE_NAME_TO_SAMPLING_RATE[device_name]
        if not _has_enough_data(track[track_name], sampling_rate):
            logger.debug(
                f"Case {int(track.caseid.iloc[0]):5,d}, track {track_name} has not enough data."
            )
            return None

    # NOTE: caseid is unique in a track, asserted at build time.
    case_id = track.caseid.iloc[0]
    static_data = cases.query(f"caseid == {case_id}")[STATIC_DATA_NAMES + ["caseid"]]

    if static_data.isna().any().any():
        return None

    return pd.merge(track, static_data, on="caseid")


def build_dataset(
    tracks_meta: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the dataset, there are three steps:
    - Download the raw data from VitalDB API based on `tracks_meta`
    - Format the timestamps
    - Filter data which does not have enough data

    Args:
        tracks_meta (pd.DataFrame): The tracks' meta-data (track URL and case ids) to
        retrieve.
        cases (pd.DataFrame): All cases information.

    Returns:
        pd.DataFrame: The dataset with all the case IDs, their track values and static
            data.
    """
    # HTTP requests handled with asynchronous calls
    tracks_raw_data = retrieve_tracks_raw_data(tracks_meta)

    # Handle timestamp
    tracks_formatted = [
        format_time_track_raw_data(track_raw_data) for track_raw_data in tracks_raw_data
    ]

    # Our post process, specific to select our tracks of interest
    # tracks_post_processed = [post_process_track(track) for track in tracks_formatted]
    # WARNING: Track info with meaning about the signal has not been, is it useful?
    logger.debug("Start post process data, keep cases with enough data")
    tracks_post_processed = [
        track
        for _track in tracks_formatted
        if (track := post_process_track(_track, cases)) is not None
    ]
    logger.debug("Post post process is done.")

    # On each case dataframe, count the number of tracks.
    all_track_names = set(get_track_names())
    n_tracks = sum(
        len(all_track_names & set(track.columns)) for track in tracks_post_processed
    )
    logger.info(
        f"Dataset succesfully built with {len(tracks_post_processed):,d} cases "
        f"({n_tracks:,d} tracks)."
    )

    return pd.concat(
        [
            track_post_processed
            for track_post_processed in tracks_post_processed
            if track_post_processed is not None
        ]
    )


def main():
    # Get args and set logger
    log_level_name, output_folder = parse()
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    setup_logger(output_folder, log_level_name)

    logger.debug("Start retrieving track meta data and cases CSV from VitalDB.")
    tracks_meta = pd.read_csv(TRACKS_META_URL, dtype={"tname": "category"})
    cases = pd.read_csv(CASE_INFO_URL)
    logger.debug("Done retrieving track meta data and cases CSV from VitalDB.")

    case_ids = filter_case_ids(cases, tracks_meta)
    logger.info(f"Number of cases to consider: {len(case_ids):,d}\n")

    track_names = get_track_names()
    targeted_tracks_meta = tracks_meta[
        tracks_meta.tname.isin(track_names) & tracks_meta.caseid.isin(case_ids)
    ]
    logger.info(f"Number of tracks to download: {len(targeted_tracks_meta):,d}\n")

    dataset = build_dataset(targeted_tracks_meta, cases)

    dataset_file = output_folder / "data_async.csv"
    logger.info(f"Dataset is saved in {dataset_file}.")
    dataset.to_csv(dataset_file)


if __name__ == "__main__":
    main()
