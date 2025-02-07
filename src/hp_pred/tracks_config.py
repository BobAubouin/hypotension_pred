import numpy as np
from typing import TypedDict

STATIC_DATA_NAMES = ["age", "bmi", "asa", "preop_cr", "preop_htn", "opname"]
STATIC_NAME_TO_DTYPES = {
    "age": np.uint16,
    "bmi": np.float16,
    "preop_cr": np.float32,
    "asa": np.uint16,
    "preop_htn": np.uint16,
    "opname": "category",
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
    TrackConfig(name="Orchestra", tracks=["PPF20_CT", "RFTN20_CT"]),
    TrackConfig(name="Primus", tracks=["MAC"]),
]

DEVICE_NAME_TO_SAMPLING_RATE = {
    "Solar8000": 2,
    "Primus": 7,
    "BIS": 1,
}

WAV_TRACK = [
    'SNUADC/*',
    'Primus/AWP',
    'Primus/CO2',
    'BIS/EEG1_WAV',
    'BIS/EEG2_WAV']
