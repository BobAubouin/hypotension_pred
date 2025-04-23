import numpy as np
import pandas as pd
from typing import TypedDict

STATIC_DATA_NAMES = ["age", "bmi", "asa"]
STATIC_NAME_TO_DTYPES = {
    "age": np.uint16,
    "bmi": np.float16,
    # "preop_cr": np.float32,
    "asa": np.uint16,
    # "preop_htn": np.uint16,
    # "opname": "category",
}

SAMPLING_TIME = 2


class TrackConfig(TypedDict):
    name: str
    tracks: list[str]


TRACKS_CONFIG = [
    TrackConfig(
        name="Solar8000",
        tracks=[
            "ART_MBP",
            "ART_SBP",
            "ART_DBP",
            "HR",
            "RR_CO2",
            "PLETH_SPO2",
            "ETCO2",
            "BT",
        ],
    ),
    TrackConfig(
        name="Orchestra",
        tracks=["PPF20_CT",
                "RFTN20_CT",
                "VASO_RATE",
                "PHEN_RATE",
                "NEPI_RATE",
                "EPI_RATE",
                "DOPA_RATE",
                "DOBU_RATE",
                "DTZ_RATE",
                "NTG_RATE",
                "NPS_RATE",
                ],
    ),
    TrackConfig(name="Primus", tracks=["MAC"]),
    TrackConfig(name="SNUADC", tracks=["ART"]),
    TrackConfig(name="BIS", tracks=["BIS"]),
]

DEVICE_NAME_TO_SAMPLING_RATE = {
    "Solar8000": 2,
    "Primus": 7,
    "BIS": 1,
}
TRACK_NAME_MBP = "Solar8000/ART_MBP"
TRACK_NAME_PPF = "Orchestra/PPF20_CT"
CASEEND_CASE_THRESHOLD = 3600  # seconds
FORBIDDEN_OPNAME_CASE = "transplant"
AGE_CASE_THRESHOLD = 18  # years
BLOOD_LOSS_THRESHOLD = 200  # mL
BOLUS_THRESHOLD_EPH = 9  # mg
BOLUS_THRESHOLD_PHE = 500  # mcg
BOLUS_THRESHOLD_EPI = 0  # mcg
BOLUS_THRESHOLD_FTN = 50  # mcg


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
    # The cases should have the Mean Blood Pressure track.
    cases_with_mbp = tracks_meta.query(f"tname == '{TRACK_NAME_MBP}'").caseid.unique()
    cases_with_ppf_tci = tracks_meta.query(f"tname == '{TRACK_NAME_PPF}'").caseid.unique()

    case_with_ppf_not_induction = cases[
        (cases.intraop_ppf > 3 * cases.weight)
        | (
            (cases.intraop_ppf > 0)
            & (cases.caseid.isin(cases_with_ppf_tci))
        )
    ]
    # The cases should met these requirements
    filtered_unique_case_ids = cases[
        (cases.caseid.isin(cases_with_mbp))
        & (~cases.caseid.isin(case_with_ppf_not_induction.caseid))
        & (cases.age > AGE_CASE_THRESHOLD)
        & (cases.caseend > CASEEND_CASE_THRESHOLD)
        & (~cases.opname.str.contains(FORBIDDEN_OPNAME_CASE, case=False))
        & (~cases.optype.str.contains(FORBIDDEN_OPNAME_CASE, case=False))
        & (cases.intraop_eph <= BOLUS_THRESHOLD_EPH)
        & (cases.intraop_phe <= BOLUS_THRESHOLD_PHE)
        & (cases.intraop_epi <= BOLUS_THRESHOLD_EPI)
        & (cases.intraop_ftn <= BOLUS_THRESHOLD_FTN)
        & (cases.emop == 0)
        & (
            (cases.intraop_ebl < BLOOD_LOSS_THRESHOLD)
            | (cases.intraop_ebl.isna())
        )
    ].caseid.unique()

    # specific case for propofol

    # The cases should have the needed static data
    potential_cases = cases[cases.caseid.isin(filtered_unique_case_ids)]
    filtered_case_ids = potential_cases[
        potential_cases[STATIC_DATA_NAMES + ["caseid"]].isna().sum("columns") == 0
    ].caseid.tolist()

    return filtered_case_ids
