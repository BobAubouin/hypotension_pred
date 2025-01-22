import argparse
import asyncio
import datetime
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from hp_pred.constants import VITAL_API_BASE_URL
from hp_pred.data_retrieve_async import retrieve_tracks_raw_data_async
from hp_pred.tracks_config import (
    STATIC_DATA_NAMES,
    STATIC_NAME_TO_DTYPES,
    TRACKS_CONFIG,
    TrackConfig,
)

TRACKS_META_URL = f"{VITAL_API_BASE_URL}/trks"
CASE_INFO_URL = f"{VITAL_API_BASE_URL}/cases"

# Filter constants
TRACK_NAME_MBP = "Solar8000/ART_MBP"
# Duration in seconds
CASEEND_CASE_THRESHOLD = 3600  # seconds
FORBIDDEN_OPNAME_CASE = "transplant"
PERCENT_MISSING_DATA_THRESHOLD = 0.2
AGE_CASE_THRESHOLD = 18  # years
BLOOD_LOSS_THRESHOLD = 200  # mL
BOLUS_THRESHOLD = 20  # mg

PARQUET_SUBFOLDER_NAME = "cases"
BASE_FILENAME_DATASET = "cases_data"
BASE_FILENAME_STATIC_DATA = "static_data"


def parse() -> tuple[str, Path, int]:
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
        "-s",
        "--group_size",
        type=str,
        default=950,
        help="Amount of cases dowloaded and processed. (default: %(default)s)",
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
    group_size = int(args.group_size)

    return log_level_name, output_folder, group_size


def setup_logger(output_folder: Path, log_level: str):
    global logger
    logger = logging.getLogger("log")

    logger.setLevel(logging.DEBUG)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_formatter = logging.Formatter(log_format)

    # Console handler, log everything.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_formatter)

    # File handler
    log_file_name = output_folder / f"run-{datetime.datetime.now()}.log"
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)
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
        - Blood loss should be NaN or smaller of the threshold
        - The case should have some static data which are mandatory.

    Note: This filter is not configurable on purpose, it is meant to be static.

    Args:
        cases (pd.DataFrame): Dataframe of the VitalDB cases
        tracks_meta (pd.DataFrame): The meta-data of the cases.

    Returns:
        list[int]: List of the valid case IDs.
    """
    logger.debug("Filter case IDs: Start")
    logger.info(f"Filter case IDs: Number of cases to consider {len(cases.caseid)}")
    # The cases should have the Mean Blood Pressure track.
    cases_with_mbp = pd.merge(
        tracks_meta.query(f"tname == '{TRACK_NAME_MBP}'"),
        cases,
        on="caseid",
    )

    # The cases should met these requirements
    filtered_unique_case_ids = cases_with_mbp[
        (cases_with_mbp.age > AGE_CASE_THRESHOLD)
        & (cases_with_mbp.caseend > CASEEND_CASE_THRESHOLD)
        & (~cases_with_mbp.opname.str.contains(FORBIDDEN_OPNAME_CASE, case=False))
        & (~cases_with_mbp.optype.str.contains(FORBIDDEN_OPNAME_CASE, case=False))
        & (cases_with_mbp.emop == 0)
        & (cases_with_mbp.intraop_eph <= BOLUS_THRESHOLD)
        & (cases_with_mbp.intraop_phe <= BOLUS_THRESHOLD)
        & (cases_with_mbp.intraop_epi <= BOLUS_THRESHOLD)
        & (cases_with_mbp.intraop_mdz <= BOLUS_THRESHOLD)
        & (
            (cases_with_mbp.intraop_ebl < BLOOD_LOSS_THRESHOLD)
            | (cases_with_mbp.intraop_ebl.isna())
        )
    ].caseid.unique()

    # The cases should have the needed static data
    potential_cases = cases[cases.caseid.isin(filtered_unique_case_ids)]
    filtered_case_ids = potential_cases[
        potential_cases[STATIC_DATA_NAMES + ["caseid"]].isna().sum("columns") == 0
    ].caseid.tolist()

    n_kept_cases = len(filtered_case_ids)
    logger.info(f"Filter case IDs: Number of cases kept {n_kept_cases}")
    logger.debug("Filter case IDs: End")
    return filtered_case_ids


def retrieve_tracks_raw_data(tracks_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Use the `hp_pred.data_retrieve_async` module to get new data.
    Plus concatenate all the track, set types for track and caseid.

    Args:
        tracks_meta (pd.DataFrame): The tracks' meta-data (track URL and case ids) to
        retrieve.

    Returns:
        pd.DataFrame: The tracks data
    """
    logger.debug("Retrieve data from VitalDB API: Start")

    tracks_url_and_case_id = [
        (f"/{track.tid}", int(track.caseid))  # type: ignore
        for track in tracks_meta.itertuples(index=False)
    ]

    logger.debug("Retrieve data from VitalDB API: Start async jobs")
    tracks_raw_data = asyncio.run(
        retrieve_tracks_raw_data_async(tracks_url_and_case_id)
    )
    logger.debug("Retrieve data from VitalDB API: End async jobs")

    tracks_raw_data = pd.concat(tracks_raw_data)
    tracks_raw_data.caseid = tracks_raw_data.caseid.astype("UInt16")
    track_name_to_dtype = {
        column: "Float32"
        for column in tracks_raw_data.columns
        if column not in ["caseid", "Time"]
    }
    tracks_raw_data = tracks_raw_data.astype(track_name_to_dtype)
    logger.debug("Retrieve data from VitalDB API: Cast data types")

    logger.debug("Retrieve data from VitalDB API: End")
    return tracks_raw_data


def format_track_raw_data_wav(track_raw_data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Data formatting: Enter WAV formatting")


def format_track_raw_data_num(tracks_raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the track's raw data according to the Time column. The Time column is rounded
    and we group the different values with the same rounded Time value.

    Args:
        track_raw_data (pd.DataFrame): Raw data retrieved from the VitalDB API.

    Returns:
        pd.DataFrame: Track data with integer Time and fewer NaN.
    """
    logger.debug("Data formatting: Enter NUM formatting")
    tracks_raw_data.Time = tracks_raw_data.Time.round().astype("UInt16")
    logger.debug("Data formatting: Time is converted to pandas UInt16")

    group_columns = ["caseid", "Time"]
    aggregate_dict = {
        column: "first"  # Force agg to get the first not NaN by ("caseid", "Time")
        for column in tracks_raw_data  # Column is caseid, Time or track_name
        if column not in group_columns  # Exclude case_id and Time
    }
    tracks_raw_data_grouped = tracks_raw_data.groupby(group_columns, as_index=False)
    tracks = tracks_raw_data_grouped.agg(aggregate_dict)
    logger.debug("Data formatting: One value of Time per case ID")

    return tracks


def format_time_track_raw_data(tracks_raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the track's raw data. It chooses between the numeric and the wave formats.

    Args:
        track_raw_data (pd.DataFrame): Raw data retrieved from the VitalDB API.

    Returns:
        pd.DataFrame: Track data with integer Time and fewer NaN.
    """
    logger.debug("Data formatting: Start")

    formatted_tracks = (
        format_track_raw_data_wav(tracks_raw_data)
        if tracks_raw_data.Time.hasnans
        else format_track_raw_data_num(tracks_raw_data)
    )

    logger.debug("Data formatting: End")
    return formatted_tracks


def to_parquet(tracks: pd.DataFrame, output_folder: Path) -> None:
    """
    Create `.parquet` files from tracks, one file per case ID.

    Args:
        tracks (pd.DataFrame): Tracks data, caseid and Time
        output_folder (Path): general destination folder
    """
    logger.debug("Parquet export: Start")
    parquet_folder = output_folder / PARQUET_SUBFOLDER_NAME
    n_export = 0

    for case_id, group in tracks.groupby("caseid"):
        parquet_file = parquet_folder / f"case-{case_id:04d}.parquet"
        group.to_parquet(parquet_file, index=False)
        n_export += 1

    logger.info(f"Parquet export: {n_export:,d} files exported")
    logger.debug("Parquet export: End")


def build_dataset(
    tracks_meta: pd.DataFrame,
    cases: pd.DataFrame,
    group_size: int,
    output_folder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the dataset, there are three steps:
    - Download the raw data from VitalDB API based on `tracks_meta`
    - Format the timestamps

    Args:
        tracks_meta (pd.DataFrame): The tracks' meta-data (track URL and case ids) to
        retrieve.
        cases (pd.DataFrame): All cases information.

    Returns:
        pd.DataFrame: The dataset with all the case IDs (track time series)
        pd.DataFrame: Static data for each case
    """
    logger.debug("Build dataset: Start")
    case_ids = filter_case_ids(cases, tracks_meta)

    track_names = get_track_names()
    tracks_meta = tracks_meta[
        tracks_meta.tname.isin(track_names) & tracks_meta.caseid.isin(case_ids)
    ]
    logger.info(f"Buid dataset: Number of tracks to download {len(tracks_meta):,d}")

    n_case_ids = len(case_ids)
    case_ids_groups = [
        case_ids[i: i + group_size] for i in range(0, n_case_ids, group_size)
    ]
    logger.info(
        f"Buid dataset: Group size {group_size}, "
        f"Number of groups {len(case_ids_groups)}"
    )

    (output_folder / PARQUET_SUBFOLDER_NAME).mkdir(exist_ok=True)
    for i, case_ids_group in enumerate(case_ids_groups):
        logger.debug(f"Buid dataset: Group {i}")
        tracks_meta_group = tracks_meta[tracks_meta.caseid.isin(case_ids_group)]

        # HTTP requests handled with asynchronous calls
        tracks_raw_data = retrieve_tracks_raw_data(tracks_meta_group)

        # Handle timestamp, index
        tracks = format_time_track_raw_data(tracks_raw_data)

        # To parquet files
        to_parquet(tracks, output_folder)

    del tracks_meta, tracks
    tracks = pd.read_parquet((output_folder / PARQUET_SUBFOLDER_NAME))
    logger.debug("Build dataset: Load the whole dataset")

    cases = cases[cases.caseid.isin(case_ids)]
    static_data = cases[STATIC_DATA_NAMES + ["caseid"]].astype(STATIC_NAME_TO_DTYPES)
    logger.debug("Build dataset: Static data created")

    logger.debug("Build dataset: End")
    return tracks, static_data


def main():
    # Get args and set logger
    log_level_name, output_folder, group_size = parse()
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    setup_logger(output_folder, log_level_name)
    logger.info("The use of data from VitalDB is subject to the terms of use, see: https://vitaldb.net/dataset/#h.vcpgs1yemdb5")
    logger.debug("Retrieve meta data and cases data from VitalDB: Start")
    tracks_meta = pd.read_csv(TRACKS_META_URL, dtype={"tname": "category"})
    cases = pd.read_csv(CASE_INFO_URL)
    logger.debug("Retrieve meta data and cases data from VitalDB: End")

    dataset, static_data = build_dataset(tracks_meta, cases, group_size, output_folder)

    dataset_file = output_folder / f"{BASE_FILENAME_DATASET}.parquet"
    dataset.to_parquet(dataset_file, index=False)

    static_data_file = output_folder / f"{BASE_FILENAME_STATIC_DATA}.parquet"
    static_data.to_parquet(static_data_file, index=False)


if __name__ == "__main__":
    main()
