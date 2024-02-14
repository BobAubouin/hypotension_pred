import multiprocessing as mp
from functools import partial
import pandas as pd
import vitaldb as vdb
from tqdm import tqdm

PERCENT_MISSING_DATA_THRESHOLD = 0.4
AGE_CASE_THRESHOLD = 18
# Duration in seconds
CASEEND_CASE_THRESHOLD = 3600
FORBIDDEN_OPNAME_CASE = "transplant"

TRACK_LIST_URL = "https://api.vitaldb.net/trks"
CASE_INFO_URL = "https://api.vitaldb.net/cases"

TRACK_NAME_SOLAR = "Solar8000/ART_MBP"

STATIC_DATA_NAMES = ["age", "bmi", "asa", "preop_cr", "preop_htn", "opname"]

SOLAR_DEVICE_NAME = "Solar8000"
SOLAR_SIGNAL_NAMES = [
    "ART_MBP",
    "ART_SBP",
    "ART_DBP",
    "HR",
    "RR",
    "PLETH_SPO2",
    "ETCO2",
]
ORCHESTRA_DEVICE_NAME = "Orchestra"
ORCHESTRA_SIGNAL_NAMES = ["PPF20_CT"]
PRIMUS_DEVICE_NAME = "Primus"
PRIMUS_SIGNAL_NAMES = ["MAC"]

# All in seconds
SAMPLING_TIME = 1
PRIMUS_SAMPLING_TIME = 7
SOLAR_SAMPLING_TIME = 2


def download_one_case(
    caseid: int,
    signal_names: list[str],
    cases: pd.DataFrame,
) -> pd.DataFrame | None:
    """Downloads the data from the vitaldb for a single case and returns it.
    Might return None if the data does not pass the filter.

    Parameters
    ----------
    caseid : int
        Case id to download.
    signal_names : list[str]
        signal names to download.
    cases : pd.DataFrame
        dataframe containing the cases information.

    Returns
    -------
    pd.DataFrame
        Returns the data in a pandas dataframe or None if empty.
    """
    data_patient = vdb.VitalFile(caseid, signal_names)
    data_patient = data_patient.to_pandas(signal_names, SAMPLING_TIME)
    data_patient.insert(0, "caseid", caseid)

    # check if there is enough data for each signal
    for signal in signal_names:
        if SOLAR_DEVICE_NAME in signal:
            device_sampling_time = SOLAR_SAMPLING_TIME
        elif ORCHESTRA_DEVICE_NAME in signal:  # ok if missing propofol data
            continue
        elif PRIMUS_DEVICE_NAME in signal:
            device_sampling_time = PRIMUS_SAMPLING_TIME
        sampling_rate = SAMPLING_TIME / device_sampling_time

        # if more than PERCENT_MISSING_DATA_THRESHOLD of the data is missing,
        # -> skip the case
        has_not_enough_data = data_patient[signal].isna().sum() > len(data_patient) * (
            1 - sampling_rate
        ) * (1 + PERCENT_MISSING_DATA_THRESHOLD)
        if has_not_enough_data:
            return None

    # get static data frome our case
    data_patient_static = cases.query(f"caseid == {caseid}")[STATIC_DATA_NAMES]
    data_patient_static["caseid"] = caseid

    if data_patient_static.isna().any().any():
        return None

    # merge the data
    data_patient = pd.merge(data_patient, data_patient_static, on="caseid")
    return data_patient


def download_dataset(signal_names: list[str]) -> pd.DataFrame:
    """Downloads the data from the vitaldb and returns the data in a pandas
    dataframe.

    Parameters
    ----------
    signal_names : list[str]
        Signal names to download.

    Returns
    -------
    pd.DataFrame
        Columns of the output dataframe are the caseid + signal names +
        static_data.
    """
    tracks = pd.read_csv(TRACK_LIST_URL)
    cases = pd.read_csv(CASE_INFO_URL)

    # select case list which fit all those conditions
    caseids = list(
        # Track name is TRACK_NAME_SOLAR.
        set(tracks.query(f"tname == '{TRACK_NAME_SOLAR}'").caseid)
        &
        # Case should have more than AGE_CASE_THRESHOLD age.
        set(cases.query(f"age > {AGE_CASE_THRESHOLD}").caseid)
        &
        # Case should have duration > CASEEND_CASE_THRESHOLD.
        set(cases.query(f"caseend > {CASEEND_CASE_THRESHOLD}").caseid)
        &
        # Case should NOT be have operator named .
        set(cases[~cases.opname.str.contains(FORBIDDEN_OPNAME_CASE, case=False)].caseid)
    )
    print(f"Number of cases to consider: {len(caseids)}")

    worker_download_function = partial(
        download_one_case,
        signal_names=signal_names,
        cases=cases,
    )

    # download the selected data
    with mp.Pool(mp.cpu_count()) as pool:
        cases_data = list(
            tqdm(
                pool.map(worker_download_function, caseids),
                total=len(caseids),
                desc="Downloading data",
            )
        )
    dataset = pd.concat(
        [case_data for case_data in cases_data if case_data is not None]
    )

    print(f"Number of cases with enough data: {len(dataset.caseid.unique())}")
    return dataset


def main():
    # signal names: <DEVICE_NAME>/<SIGNAL_NAME>
    signal_names = (
        [f"{SOLAR_DEVICE_NAME}/" + f for f in SOLAR_SIGNAL_NAMES]
        + [f"{ORCHESTRA_DEVICE_NAME}/" + f for f in ORCHESTRA_SIGNAL_NAMES]
        + [f"{PRIMUS_DEVICE_NAME}/" + f for f in PRIMUS_SIGNAL_NAMES]
    )

    dataset = download_dataset(signal_names)

    print(dataset.head())
    dataset.to_csv("data.csv", index=False)


if __name__ == "__main__":
    main()
